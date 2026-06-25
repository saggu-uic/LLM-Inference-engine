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
Speedup vs target-only ≈ (1 + k*α) / (1 + k * cost_draft/cost_target)

Models
-------
  draft  = gpt2        (117 M params)
  target = gpt2-medium (345 M params)

KV cache management (O(1) per step — the key efficiency fix)
--------------------------------------------------------------
Naive implementations re-prefill the draft model on the FULL sequence after
each acceptance step, costing O(n) draft forwards per step.

This implementation instead:
  1. _draft_k saves the intermediate KV state after each draft token (kv_states[i]
     = draft KV after seeing i draft tokens on top of the current context).
  2. After accepting n_acc draft tokens, we index kv_states[n_acc] to recover the
     KV at exactly the right prefix — no re-prefill needed.
  3. We run the draft model once on the resampled/bonus token to advance its KV.
  4. For the target, we trim the KV returned by _verify (which covers all k draft
     tokens) down to n_acc and then run target on the last token only.

Result: each speculative step costs exactly (k+1) draft calls + 2 target calls,
regardless of how long the sequence has grown.
"""
import time
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer


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
        """Return (past_kv, last_logits) for the given token sequence."""
        out = model(ids, use_cache=True)
        return out.past_key_values, out.logits[:, -1, :]

    @torch.no_grad()
    def _draft_k(self, k: int, draft_kv, draft_logits):
        """
        Generate k tokens from the draft model greedily.

        Returns:
          tokens      : list of k (1,1) token tensors
          probs       : list of k (1, vocab) probability tensors
          kv_states   : list of k+1 past_key_values snapshots
                        kv_states[i] = draft KV after i draft tokens
                        (kv_states[0] = draft_kv passed in)
          last_logits : draft logits after the k-th draft token
        """
        tokens: list[torch.Tensor] = []
        probs: list[torch.Tensor] = []
        kv_states = [draft_kv]  # kv_states[i] = KV before drafting token i
        last_logits = draft_logits

        for _ in range(k):
            next_tok = last_logits.argmax(dim=-1, keepdim=True)
            tokens.append(next_tok)
            probs.append(F.softmax(last_logits, dim=-1))
            out = self.draft(next_tok, past_key_values=kv_states[-1], use_cache=True)
            kv_states.append(out.past_key_values)
            last_logits = out.logits[:, -1, :]

        return tokens, probs, kv_states, last_logits

    @torch.no_grad()
    def _verify(self, draft_tokens, target_kv):
        """Run target on all k draft tokens in one forward pass."""
        draft_seq = torch.cat(draft_tokens, dim=-1)  # (1, k)
        out = self.target(draft_seq, past_key_values=target_kv, use_cache=True)
        return out.logits, out.past_key_values

    def _accept_reject(self, draft_tokens, draft_probs, target_logits, target_logits_prev):
        """
        Rejection sampling over k draft tokens.

        target_logits_prev: target distribution at the position BEFORE seeing d_0.
        Returns (accepted_token_list, n_accepted) where n_accepted is the count of
        draft tokens that passed (before the resampled/bonus appended at end).
        """
        eos = self.draft_tok.eos_token_id
        # Target distribution for checking each draft position:
        #   d_0 is checked against target_logits_prev (before seeing any draft)
        #   d_i (i>0) is checked against target_logits[:, i-1, :] (after d_0..d_{i-1})
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

    @staticmethod
    def _trim_kv(past_kv, length: int):
        """Trim KV cache to keep only the first `length` sequence positions."""
        return tuple(
            (k[:, :, :length, :], v[:, :, :length, :])
            for k, v in past_kv
        )

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

        # Pre-fill both models on the shared prompt
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

            # 1. Draft generates k tokens, saving intermediate KV states
            d_tokens, d_probs, draft_kv_states, _ = self._draft_k(
                k_this, draft_kv, draft_last_logits
            )

            # 2. Target verifies all k draft tokens in one forward pass
            #    t_full_kv covers: prompt + all prev-accepted tokens + k draft tokens
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

            # 4. Update KV caches — O(1) per step
            #
            # Draft: use saved state at n_acc (already covers accepted prefix),
            #        then advance one step on last_tok.
            draft_kv = draft_kv_states[n_acc]
            out_d = self.draft(last_tok, past_key_values=draft_kv, use_cache=True)
            draft_kv = out_d.past_key_values
            draft_last_logits = out_d.logits[:, -1, :]

            # Target: trim t_full_kv to exactly the accepted prefix,
            #         then advance one step on last_tok.
            #   generated.shape[1] - 1  =  context_before_step + n_acc  ✓
            trim_len = generated.shape[1] - 1
            trimmed_target_kv = self._trim_kv(t_full_kv, trim_len)
            out_t = self.target(last_tok, past_key_values=trimmed_target_kv, use_cache=True)
            target_kv = out_t.past_key_values
            target_last_logits = out_t.logits[:, -1, :]
            n_target_calls += 1  # 1 call for last_tok (was: ALL new tokens)

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
