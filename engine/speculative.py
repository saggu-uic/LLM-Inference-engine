"""
Speculative decoding (Chen et al., 2023).

Key idea
---------
A small, fast DRAFT model generates k tokens autoregressively.
The large TARGET model verifies all k in ONE forward pass.

Acceptance sampling guarantees the output distribution equals sampling
from the target alone:
  Accept draft token x with prob = min(1, P_target(x) / P_draft(x))
  On rejection at position i, resample from residual (P_T - P_D).clamp(0).
  If all k accepted, sample a bonus token from target's distribution.

Expected tokens per target call = 1 + k * α  (where α = mean accept rate)

Models: draft = gpt2 (117 M), target = gpt2-medium (345 M), k = 4.

DynamicCache handling (transformers >= 4.36)
---------------------------------------------
GPT-2 in transformers 4.36+ uses DynamicCache as past_key_values.
DynamicCache is MUTABLE — model calls update it in place via torch.cat.
torch.cat always creates a NEW tensor and replaces the list entry, so
copying the list (not the tensors) is a safe, O(1) snapshot of the current
state.  Future updates replace entries in the original list; the snapshot's
list still references the old tensors.

_snapshot_kv(): shallow-copies the key/value lists → cheap, safe snapshot.
_trim_kv():     builds a new DynamicCache with sliced tensors → for trimming
                t_full_kv back to the accepted prefix after rejection.
"""
import time
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.cache_utils import DynamicCache


# ---------------------------------------------------------------------------
# DynamicCache utilities
# ---------------------------------------------------------------------------

def _snapshot_kv(cache: DynamicCache) -> DynamicCache:
    """
    Cheap, safe snapshot of a DynamicCache.

    DynamicCache.update() replaces self.key_cache[i] with a brand-new tensor
    (torch.cat result), so copying the list captures the current tensor
    references without risk of aliasing from future updates.
    """
    snap = DynamicCache()
    snap.key_cache = list(cache.key_cache)
    snap.value_cache = list(cache.value_cache)
    if hasattr(cache, "_seen_tokens"):
        snap._seen_tokens = cache._seen_tokens
    return snap


def _trim_kv(cache: DynamicCache, length: int) -> DynamicCache:
    """Return a new DynamicCache containing only the first `length` positions."""
    trimmed = DynamicCache()
    trimmed.key_cache = [k[:, :, :length, :] for k in cache.key_cache]
    trimmed.value_cache = [v[:, :, :length, :] for v in cache.value_cache]
    if hasattr(trimmed, "_seen_tokens"):
        trimmed._seen_tokens = length
    return trimmed


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
        """Run the full prompt; return (DynamicCache, last_logits)."""
        out = model(ids, use_cache=True)
        return out.past_key_values, out.logits[:, -1, :]

    @torch.no_grad()
    def _draft_k(self, k: int, draft_kv: DynamicCache, draft_logits):
        """
        Generate k tokens from the draft model greedily.

        draft_kv is mutated in place (each model call extends it).
        We take a list snapshot BEFORE each step so we can recover
        any accepted prefix in O(1):

          kv_states[i] = state after i draft tokens
          kv_states[0] = snapshot of the incoming draft_kv (0 tokens added)
          kv_states[k] = snapshot after all k tokens
        """
        tokens: list[torch.Tensor] = []
        probs: list[torch.Tensor] = []
        # Snapshot BEFORE the first step (= state entering this call)
        kv_states: list[DynamicCache] = [_snapshot_kv(draft_kv)]
        last_logits = draft_logits

        for _ in range(k):
            next_tok = last_logits.argmax(dim=-1, keepdim=True)
            tokens.append(next_tok)
            probs.append(F.softmax(last_logits, dim=-1))
            # Mutate draft_kv with this token's KV
            out = self.draft(next_tok, past_key_values=draft_kv, use_cache=True)
            last_logits = out.logits[:, -1, :]
            # Snapshot AFTER mutation: list copy captures current tensor refs
            kv_states.append(_snapshot_kv(draft_kv))

        return tokens, probs, kv_states, last_logits

    @torch.no_grad()
    def _verify(self, draft_tokens, target_kv: DynamicCache):
        """Run target on all k draft tokens; target_kv is mutated in place."""
        draft_seq = torch.cat(draft_tokens, dim=-1)  # (1, k)
        out = self.target(draft_seq, past_key_values=target_kv, use_cache=True)
        return out.logits, out.past_key_values  # past_key_values IS target_kv, mutated

    def _accept_reject(self, draft_tokens, draft_probs, target_logits, target_logits_prev):
        """
        Rejection sampling.  Returns (accepted_list, n_accepted) where
        n_accepted counts draft tokens that passed (bonus/resample appended last).
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
            if torch.rand(1).item() <= min(1.0, p_t / max(p_d, 1e-9)):
                accepted.append(d_tok)
                n_accepted += 1
                if tok_id == eos:
                    return accepted, n_accepted
            else:
                residual = (t_dist - d_prob).clamp(min=0.0)
                s = residual.sum()
                residual = residual / s if s > 1e-9 else t_dist
                accepted.append(torch.multinomial(residual[0], 1).unsqueeze(0))
                return accepted, n_accepted

        # All k accepted → bonus token
        bonus_dist = F.softmax(target_logits[:, -1, :], dim=-1)
        accepted.append(torch.multinomial(bonus_dist[0], 1).unsqueeze(0))
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

        # Pre-fill both models on the full prompt
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

            # 1. Draft generates k tokens, mutating draft_kv; saves snapshots
            d_tokens, d_probs, draft_kv_states, _ = self._draft_k(
                k_this, draft_kv, draft_last_logits
            )

            # 2. Target verifies k draft tokens in ONE pass, mutating target_kv
            #    t_full_kv IS target_kv (same object, now extended by k tokens)
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

            # 4. O(1) KV update — no re-prefill
            #
            # Draft: kv_states[n_acc] = snapshot after exactly n_acc draft tokens.
            # Snapshot it again (fresh list copy) so the model can safely mutate it,
            # then run one forward pass to include last_tok.
            draft_kv = _snapshot_kv(draft_kv_states[n_acc])
            out_d = self.draft(last_tok, past_key_values=draft_kv, use_cache=True)
            draft_kv = out_d.past_key_values   # = draft_kv, extended by last_tok
            draft_last_logits = out_d.logits[:, -1, :]

            # Target: t_full_kv has k extra draft tokens we don't want.
            # Trim to the accepted prefix, then extend with last_tok.
            # generated.shape[1] - 1 = (context before this step) + n_acc
            trim_len = generated.shape[1] - 1
            target_kv = _trim_kv(t_full_kv, trim_len)
            out_t = self.target(last_tok, past_key_values=target_kv, use_cache=True)
            target_kv = out_t.past_key_values  # = target_kv, extended by last_tok
            target_last_logits = out_t.logits[:, -1, :]
            n_target_calls += 1  # 1 call for last_tok only (vs all new tokens before)

        total = time.perf_counter() - t0
        new_tokens = generated.shape[1] - ids.shape[1]
        text = self.target_tok.decode(generated[0], skip_special_tokens=True)

        accept_rate = n_accepted_total / max(n_draft_total, 1)
        return text, {
            "ttft_ms": ttft * 1000,
            "latency_ms": total * 1000,
            "new_tokens": new_tokens,
            "throughput_tps": new_tokens / total,
            "accept_rate": f"{accept_rate:.2%}",
            "tok_per_target_call": f"{new_tokens / max(n_target_calls, 1):.2f}",
            "n_target_calls": n_target_calls,
        }
