"""
Speculative decoding (Chen et al., 2023).

Key idea
---------
A small, fast DRAFT model generates k tokens autoregressively.
The large TARGET model then verifies all k tokens in ONE forward pass —
the same compute budget that would normally yield only 1 token.

Acceptance sampling guarantees the output distribution equals sampling
from the target alone:
  Accept draft token x with prob = min(1, P_target(x) / P_draft(x))
  On rejection at position i, resample from residual (P_T - P_D).clamp(0).
  If all k are accepted, sample a bonus token from the target's distribution.

Expected tokens per target call = 1 + k * α  (where α = mean accept rate)

Models
-------
  draft  = gpt2        (117 M params)
  target = gpt2-medium (345 M params)

KV cache management (O(1) per step)
-------------------------------------
Naive implementations re-prefill the draft model on the FULL sequence after
each acceptance step — O(n) forwards per step that grows without bound.

This implementation instead:
  1. _draft_k saves a KV snapshot before each draft token (kv_states[i] =
     KV after seeing i draft tokens on top of the current context).
  2. After accepting n_acc draft tokens, we index kv_states[n_acc] to recover
     the right prefix — no re-prefill needed.
  3. We run the draft/target each once on the resampled/bonus token.

NOTE on transformers >= 4.36:
  Models may return past_key_values as a DynamicCache object rather than a
  plain tuple-of-tuples. DynamicCache is mutable — the same object is updated
  in place on each model call — so saving it directly into kv_states would
  corrupt earlier snapshots. We therefore call _to_tuple() immediately after
  every model call to convert to an immutable tuple-of-(key,value) snapshot.
"""
import time
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# ---------------------------------------------------------------------------
# KV cache format utilities
# ---------------------------------------------------------------------------

def _to_tuple(past_kv):
    """
    Normalize past_key_values to an immutable tuple-of-(key, value) per layer.

    Handles both:
      - Legacy format: tuple of (key, value) pairs (one per layer)
      - DynamicCache  (transformers >= 4.36): has .key_cache / .value_cache lists
        that are mutated in-place across calls — we snapshot them here.
    """
    if past_kv is None:
        return None
    if hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache"):
        # DynamicCache: zip the per-layer key/value lists into 2-tuples.
        # Each k/v tensor is created fresh by torch.cat inside DynamicCache.update(),
        # so the snapshot is safe even after subsequent model calls.
        return tuple(zip(past_kv.key_cache, past_kv.value_cache))
    # Already a tuple of tuples — return as-is (no mutation risk).
    return tuple(past_kv)


def _trim_kv(past_kv, length: int):
    """
    Trim past_key_values so only the first `length` positions are retained.

    `past_kv` must be in the normalized tuple-of-(key,value) format
    produced by _to_tuple().
    """
    return tuple(
        (k[:, :, :length, :], v[:, :, :length, :])
        for k, v in past_kv
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SpeculativeEngine:
    def __init__(
        self,
        draft_name: str = "gpt2",
        target_name: str = "gpt2-medium",
        device: str | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"  Loading draft  : {draft_name}")
        self.draft_tok = GPT2Tokenizer.from_pretrained(draft_name)
        self.draft = GPT2LMHeadModel.from_pretrained(draft_name, dtype=torch.float32)
        self.draft.to(self.device).eval()

        print(f"  Loading target : {target_name}")
        self.target_tok = GPT2Tokenizer.from_pretrained(target_name)
        self.target = GPT2LMHeadModel.from_pretrained(target_name, dtype=torch.float32)
        self.target.to(self.device).eval()

        assert self.draft_tok.vocab_size == self.target_tok.vocab_size

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _prefill(self, model, ids):
        """Return (past_kv_tuple, last_logits) for the given token sequence."""
        out = model(ids, use_cache=True)
        return _to_tuple(out.past_key_values), out.logits[:, -1, :]

    @torch.no_grad()
    def _draft_k(self, k: int, draft_kv, draft_logits):
        """
        Generate k tokens from the draft model greedily.

        Returns:
          tokens      : list of k (1,1) token tensors
          probs       : list of k (1, vocab) probability tensors
          kv_states   : list of k+1 normalized KV snapshots
                        kv_states[i] = draft KV after i draft tokens
          last_logits : draft logits after the k-th draft token
        """
        tokens: list[torch.Tensor] = []
        probs: list[torch.Tensor] = []
        kv_states = [draft_kv]  # kv_states[0] = KV before any draft token
        last_logits = draft_logits

        for _ in range(k):
            next_tok = last_logits.argmax(dim=-1, keepdim=True)
            tokens.append(next_tok)
            probs.append(F.softmax(last_logits, dim=-1))
            out = self.draft(next_tok, past_key_values=kv_states[-1], use_cache=True)
            # Snapshot immediately — _to_tuple() prevents DynamicCache aliasing
            kv_states.append(_to_tuple(out.past_key_values))
            last_logits = out.logits[:, -1, :]

        return tokens, probs, kv_states, last_logits

    @torch.no_grad()
    def _verify(self, draft_tokens, target_kv):
        """Run target on all k draft tokens in one forward pass."""
        draft_seq = torch.cat(draft_tokens, dim=-1)  # (1, k)
        out = self.target(draft_seq, past_key_values=target_kv, use_cache=True)
        return out.logits, _to_tuple(out.past_key_values)

    def _accept_reject(self, draft_tokens, draft_probs, target_logits, target_logits_prev):
        """
        Rejection sampling over k draft tokens.

        target_logits_prev: target distribution BEFORE seeing any draft token
                            (used to check d_0).
        Returns (accepted_token_list, n_accepted) where n_accepted = count of
        draft tokens that passed acceptance before the bonus/resample appended at end.
        """
        eos = self.draft_tok.eos_token_id
        target_dists = [F.softmax(target_logits_prev, dim=-1)] + [
            F.softmax(target_logits[:, i, :], dim=-1)
            for i in range(len(draft_tokens) - 1)
        ]

        accepted: list[torch.Tensor] = []
        n_accepted = 0

        for d_tok, d_prob, t_dist in zip(draft_tokens, draft_probs, target_dists):
            tok_id = d_tok.item()
            p_t = t_dist[0, tok_id].item()
            p_d = d_prob[0, tok_id].item()
            accept_p = min(1.0, p_t / max(p_d, 1e-9))

            if torch.rand(1).item() <= accept_p:
                accepted.append(d_tok)
                n_accepted += 1
                if tok_id == eos:
                    return accepted, n_accepted
            else:
                residual = (t_dist - d_prob).clamp(min=0.0)
                s = residual.sum()
                residual = residual / s if s > 1e-9 else t_dist
                new_tok = torch.multinomial(residual[0], 1).unsqueeze(0)
                accepted.append(new_tok)
                return accepted, n_accepted

        # All k accepted → bonus token from target's distribution at position k
        bonus_dist = F.softmax(target_logits[:, -1, :], dim=-1)
        bonus_tok = torch.multinomial(bonus_dist[0], 1).unsqueeze(0)
        accepted.append(bonus_tok)
        return accepted, n_accepted

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        k: int = 4,
    ) -> tuple[str, dict]:
        ids = self.target_tok.encode(prompt, return_tensors="pt").to(self.device)
        t0 = time.perf_counter()

        # Pre-fill both models; _prefill calls _to_tuple() internally
        target_kv, target_last_logits = self._prefill(self.target, ids)
        draft_kv, draft_last_logits = self._prefill(self.draft, ids)

        ttft = time.perf_counter() - t0
        generated = ids.clone()

        n_accepted_total = 0
        n_draft_total = 0
        n_target_calls = 0
        eos = self.target_tok.eos_token_id

        while generated.shape[1] - ids.shape[1] < max_new_tokens:
            remaining = max_new_tokens - (generated.shape[1] - ids.shape[1])
            k_this = min(k, remaining)

            # 1. Draft generates k tokens; saves immutable KV snapshots per step
            d_tokens, d_probs, draft_kv_states, _ = self._draft_k(
                k_this, draft_kv, draft_last_logits
            )

            # 2. Target verifies all k draft tokens in ONE forward pass
            #    t_full_kv covers: context + k draft tokens
            t_logits, t_full_kv = self._verify(d_tokens, target_kv)
            n_target_calls += 1
            n_draft_total += k_this

            # 3. Acceptance sampling
            accepted, n_acc = self._accept_reject(
                d_tokens, d_probs, t_logits, target_last_logits
            )
            n_accepted_total += n_acc
            last_tok = accepted[-1]  # resampled or bonus token

            new_ids = torch.cat(accepted, dim=-1)
            generated = torch.cat([generated, new_ids], dim=-1)

            if any(t.item() == eos for t in accepted):
                break

            # 4. Update KV caches — O(1) per step, not O(n)
            #
            # Draft: kv_states[n_acc] is the snapshot AFTER n_acc draft tokens
            #        (the correct prefix). Run draft once on last_tok to advance.
            draft_kv = draft_kv_states[n_acc]
            out_d = self.draft(last_tok, past_key_values=draft_kv, use_cache=True)
            draft_kv = _to_tuple(out_d.past_key_values)
            draft_last_logits = out_d.logits[:, -1, :]

            # Target: trim t_full_kv to only the accepted prefix, then run on last_tok.
            #   generated.shape[1] - 1 = context_before_this_step + n_acc
            trim_len = generated.shape[1] - 1
            trimmed_target_kv = _trim_kv(t_full_kv, trim_len)
            out_t = self.target(last_tok, past_key_values=trimmed_target_kv, use_cache=True)
            target_kv = _to_tuple(out_t.past_key_values)
            target_last_logits = out_t.logits[:, -1, :]
            n_target_calls += 1  # 1 call for last_tok only (vs ALL new tokens before)

        total = time.perf_counter() - t0
        new_tokens = generated.shape[1] - ids.shape[1]
        text = self.target_tok.decode(generated[0], skip_special_tokens=True)

        accept_rate = n_accepted_total / max(n_draft_total, 1)
        tokens_per_target_call = new_tokens / max(n_target_calls, 1)

        return text, {
            "ttft_ms": ttft * 1000,
            "latency_ms": total * 1000,
            "new_tokens": new_tokens,
            "throughput_tps": new_tokens / total,
            "accept_rate": f"{accept_rate:.2%}",
            "tok_per_target_call": f"{tokens_per_target_call:.2f}",
            "n_target_calls": n_target_calls,
        }
