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

KV cache strategy
------------------
transformers returns past_key_values as a DynamicCache.  We never touch its
internal attributes — we only use two public operations:
  * model(tokens, past_key_values=cache)  advances the cache in place
  * cache.crop(length)                    trims the cache back to `length`

Per-step flow (this is the whole point of speculative decoding):
  1. Draft proposes k tokens, advancing draft_kv from L → L+k.
  2. Verify: ONE target forward pass over all k draft tokens at once,
     advancing target_kv from L → L+k and giving logits for every position.
  3. Acceptance sampling keeps the first n_acc tokens + 1 resampled/bonus token.
  4. crop() both caches back to L+n_acc (discarding rejected positions whose KV
     was already computed in step 2), then one forward pass commits last_tok.

Cost per step:  k+1 draft calls  +  2 target calls  (verify + last_tok).
The k draft tokens are verified in a SINGLE target pass — that is what lets the
target run fewer times than the number of tokens produced.  Nothing here scales
with sequence length.
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

    def _sync(self):
        """Block until pending CUDA kernels finish so timings are real.
        On GPU, kernel launches are async; without this perf_counter would
        measure launch time, not compute.  No-op on CPU."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    @torch.no_grad()
    def _prefill(self, model, ids):
        """Run full prompt; return (past_kv, last_logits)."""
        out = model(ids, use_cache=True)
        return out.past_key_values, out.logits[:, -1, :]

    @torch.no_grad()
    def _draft_k(self, k: int, draft_kv, draft_logits):
        """
        Propose k tokens from the draft model, advancing draft_kv in place
        from length L to L+k.  Returns (tokens, probs, advanced_draft_kv).
        """
        tokens: list[torch.Tensor] = []
        probs: list[torch.Tensor] = []
        last_logits = draft_logits
        for _ in range(k):
            next_tok = last_logits.argmax(dim=-1, keepdim=True)
            tokens.append(next_tok)
            probs.append(F.softmax(last_logits, dim=-1))
            out = self.draft(next_tok, past_key_values=draft_kv, use_cache=True)
            draft_kv = out.past_key_values
            last_logits = out.logits[:, -1, :]
        return tokens, probs, draft_kv

    @torch.no_grad()
    def _verify(self, draft_tokens, target_kv):
        """
        Verify all k draft tokens in ONE target forward pass, advancing
        target_kv in place from L to L+k.  Returns (logits, advanced_target_kv).
        """
        draft_seq = torch.cat(draft_tokens, dim=-1)  # (1, k)
        out = self.target(draft_seq, past_key_values=target_kv, use_cache=True)
        return out.logits, out.past_key_values

    def _accept_reject(self, draft_tokens, draft_probs, target_logits, target_logits_prev):
        """
        Rejection sampling.  Returns (accepted_list, n_accepted) where
        n_accepted = count of draft tokens that passed acceptance.
        The resampled or bonus token is always appended last.
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

        # All k accepted → bonus token from target's distribution at position k
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
        self._sync()
        t0 = time.perf_counter()

        # Pre-fill both models on the shared prompt.
        # L = committed length: prompt + tokens accepted so far.
        target_kv, target_last_logits = self._prefill(self.target, ids)
        draft_kv, draft_last_logits = self._prefill(self.draft, ids)
        L = ids.shape[1]

        self._sync()
        ttft = time.perf_counter() - t0
        generated = ids.clone()

        n_accepted_total = 0
        n_draft_total = 0
        n_target_calls = 0
        eos = self.target_tok.eos_token_id

        while generated.shape[1] - ids.shape[1] < max_new_tokens:
            remaining = max_new_tokens - (generated.shape[1] - ids.shape[1])
            k_this = min(k, remaining)

            # 1. Draft proposes k tokens — advances draft_kv L → L+k
            d_tokens, d_probs, draft_kv = self._draft_k(k_this, draft_kv, draft_last_logits)
            n_draft_total += k_this

            # 2. Verify all k in ONE target pass — advances target_kv L → L+k
            t_logits, target_kv = self._verify(d_tokens, target_kv)
            n_target_calls += 1

            # 3. Acceptance sampling
            accepted, n_acc = self._accept_reject(
                d_tokens, d_probs, t_logits, target_last_logits
            )
            n_accepted_total += n_acc
            last_tok = accepted[-1]

            new_ids = torch.cat(accepted, dim=-1)
            generated = torch.cat([generated, new_ids], dim=-1)

            if any(t.item() == eos for t in accepted):
                break

            # 4. Commit: crop both caches to the accepted prefix (L+n_acc), then
            #    one forward pass each to add last_tok's KV and next-step logits.
            #    The verify pass already computed KV for all k positions, so the
            #    accepted ones are reused for free; only last_tok needs a forward.
            target_kv.crop(L + n_acc)
            draft_kv.crop(L + n_acc)

            ot = self.target(last_tok, past_key_values=target_kv, use_cache=True)
            target_kv = ot.past_key_values
            target_last_logits = ot.logits[:, -1, :]
            n_target_calls += 1

            od = self.draft(last_tok, past_key_values=draft_kv, use_cache=True)
            draft_kv = od.past_key_values
            draft_last_logits = od.logits[:, -1, :]

            L += n_acc + 1

        self._sync()
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
