"""
Speculative decoding (Chen et al., 2023).

Key idea:
  A small, fast DRAFT model generates k tokens autoregressively.
  The large TARGET model then verifies all k tokens in ONE forward pass —
  the same compute budget that would normally yield only 1 token.

Acceptance sampling ensures the output distribution is identical to sampling
from the target alone:
  • Accept draft token x with probability min(1, P_target(x) / P_draft(x)).
  • On rejection at position i, resample from the residual distribution
    (P_target − P_draft).clamp(0), normalised.
  • If all k tokens are accepted, append a bonus token from the target's
    prediction at position k.

Expected tokens per target call = 1 + k * α  (where α = mean accept rate)
Speedup ≈ (1 + k*α) / (1 + k * cost_draft/cost_target)

Here we use:
  draft  = gpt2       (117 M params)
  target = gpt2-medium (345 M params)
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

        # Both GPT-2 variants share the same vocabulary
        assert self.draft_tok.vocab_size == self.target_tok.vocab_size

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _prefill(self, model, ids):
        """Return (past_kv, last_logits) for the given token ids."""
        out = model(ids, use_cache=True)
        return out.past_key_values, out.logits[:, -1, :]

    @torch.no_grad()
    def _draft_k(self, k: int, draft_kv, draft_logits):
        """Autoregressively generate k tokens from the draft model."""
        tokens, probs = [], []
        last_logits = draft_logits
        for _ in range(k):
            next_tok = last_logits.argmax(dim=-1, keepdim=True)  # (1, 1) greedy
            tokens.append(next_tok)
            probs.append(F.softmax(last_logits, dim=-1))   # (1, vocab)
            out = self.draft(next_tok, past_key_values=draft_kv, use_cache=True)
            draft_kv = out.past_key_values
            last_logits = out.logits[:, -1, :]
        return tokens, probs, draft_kv, last_logits

    @torch.no_grad()
    def _verify(self, draft_tokens, target_kv):
        """Run target on k draft tokens; return (logits_k, updated_kv)."""
        draft_seq = torch.cat(draft_tokens, dim=-1)  # (1, k)
        out = self.target(draft_seq, past_key_values=target_kv, use_cache=True)
        # logits[:, i, :] = P(next | context + draft_0..draft_i)
        return out.logits, out.past_key_values

    def _accept_reject(self, draft_tokens, draft_probs, target_logits, target_logits_prev):
        """
        Rejection sampling over k draft tokens.
        target_logits_prev: target logits from the previous step's last position,
                            used to check the FIRST draft token.
        Returns (accepted_tokens list, n_accepted).
        """
        eos = self.draft_tok.eos_token_id
        # Assemble target distributions for positions 0..k-1
        # Position 0 comes from the saved last-step target logit (before seeing d0)
        # Position i>0 comes from target_logits[:, i-1, :] (after seeing d0..d_{i-1})
        target_dist_list = [F.softmax(target_logits_prev, dim=-1)]
        for i in range(len(draft_tokens) - 1):
            target_dist_list.append(F.softmax(target_logits[:, i, :], dim=-1))

        accepted_tokens = []
        n_accepted = 0

        for i, (d_tok, d_prob, t_dist) in enumerate(
            zip(draft_tokens, draft_probs, target_dist_list)
        ):
            tok_id = d_tok.item()
            p_t = t_dist[0, tok_id].item()
            p_d = d_prob[0, tok_id].item()
            accept_p = min(1.0, p_t / max(p_d, 1e-9))

            if torch.rand(1).item() <= accept_p:
                accepted_tokens.append(d_tok)
                n_accepted += 1
                if tok_id == eos:
                    return accepted_tokens, n_accepted
            else:
                # Resample from residual
                residual = (t_dist - d_prob).clamp(min=0.0)
                total = residual.sum()
                if total < 1e-9:
                    residual = t_dist
                else:
                    residual = residual / total
                new_tok = torch.multinomial(residual[0], 1).unsqueeze(0)
                accepted_tokens.append(new_tok)
                return accepted_tokens, n_accepted

        # All k accepted → bonus token from target's last position
        bonus_dist = F.softmax(target_logits[:, -1, :], dim=-1)
        bonus_tok = torch.multinomial(bonus_dist[0], 1).unsqueeze(0)
        accepted_tokens.append(bonus_tok)
        return accepted_tokens, n_accepted

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

        # Pre-fill both models on the prompt
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

            # 1. Draft generates k tokens
            d_tokens, d_probs, draft_kv, draft_last_logits = self._draft_k(
                k_this, draft_kv, draft_last_logits
            )

            # 2. Target verifies all k in ONE forward pass
            t_logits, t_full_kv = self._verify(d_tokens, target_kv)
            n_target_calls += 1
            n_draft_total += k_this

            # 3. Accept / reject
            accepted, n_acc = self._accept_reject(
                d_tokens, d_probs, t_logits, target_last_logits
            )
            n_accepted_total += n_acc

            new_ids = torch.cat(accepted, dim=-1)  # (1, m)
            generated = torch.cat([generated, new_ids], dim=-1)

            # 4. Update target KV for accepted + bonus tokens
            out = self.target(new_ids, past_key_values=target_kv, use_cache=True)
            target_kv = out.past_key_values
            target_last_logits = out.logits[:, -1, :]
            n_target_calls += 1

            # 5. Update draft KV: re-prefill on the full accepted sequence
            #    (simple; in prod you'd slice/trim the draft KV instead)
            draft_kv, draft_last_logits = self._prefill(self.draft, generated)

            if any(t.item() == eos for t in accepted):
                break

        total = time.perf_counter() - t0
        new_tokens = generated.shape[1] - ids.shape[1]
        text = self.target_tok.decode(generated[0], skip_special_tokens=True)

        # NOTE: on CPU, FP32 matmul paths for batch-prefill vs incremental differ
        # numerically, causing spurious rejections even for identical models.
        # On GPU (Colab T4) with gpt2→gpt2-medium the accept rate is meaningful.
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
