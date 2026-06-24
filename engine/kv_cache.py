"""
KV Cache inference with prefix reuse.

Two-phase decoding:
  Pre-fill  — run the full prompt through the model ONCE, store K and V tensors
              for every layer.  Cost: O(n) attention operations.
  Decode    — each new token only needs to attend to the stored KVs.
              Every step is O(1) extra attention work regardless of history length.

Prefix reuse:
  Many requests share a common system prompt.  We store the KV for that prefix
  so subsequent requests skip the pre-fill for the shared part entirely.
"""
import hashlib
import time
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Type alias: one entry per transformer layer → (key, value) tensors
KVCache = tuple[tuple[torch.Tensor, torch.Tensor], ...]


class KVCacheEngine:
    def __init__(self, model_name: str = "gpt2", device: str | None = None):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name, dtype=torch.float32)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device).eval()

        # prefix_hash → (past_key_values, prefix_token_count)
        self._prefix_store: dict[str, tuple[KVCache, int]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _token_hash(self, ids: torch.Tensor) -> str:
        return hashlib.md5(ids.cpu().numpy().tobytes()).hexdigest()

    @torch.no_grad()
    def _prefill(self, input_ids: torch.Tensor) -> tuple[KVCache, torch.Tensor]:
        """Run one forward pass over the full prompt; return KV cache + last logits."""
        out = self.model(input_ids, use_cache=True)
        return out.past_key_values, out.logits[:, -1, :]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cache_prefix(self, prefix: str) -> str:
        """Pre-compute and store KV for a shared prefix (e.g., system prompt)."""
        ids = self.tokenizer.encode(prefix, return_tensors="pt").to(self.device)
        key = self._token_hash(ids)
        if key not in self._prefix_store:
            past_kv, _ = self._prefill(ids)
            self._prefix_store[key] = (past_kv, ids.shape[1])
        return key

    def clear_prefix_cache(self):
        self._prefix_store.clear()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        greedy: bool = True,
        temperature: float = 1.0,
        shared_prefix: str | None = None,
    ) -> tuple[str, dict]:
        """
        Generate with KV cache.

        If shared_prefix is provided (must match a string passed to cache_prefix()),
        its KV is reused — the model never re-processes those tokens.
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        t0 = time.perf_counter()

        prefix_hit = False
        past_kv: KVCache | None = None

        if shared_prefix is not None:
            pfx_ids = self.tokenizer.encode(shared_prefix, return_tensors="pt").to(self.device)
            pfx_key = self._token_hash(pfx_ids)
            if pfx_key in self._prefix_store:
                past_kv, pfx_len = self._prefix_store[pfx_key]
                # Only process the tokens AFTER the shared prefix
                suffix_ids = input_ids[:, pfx_len:]
                if suffix_ids.shape[1] > 0:
                    out = self.model(suffix_ids, past_key_values=past_kv, use_cache=True)
                    past_kv = out.past_key_values
                    last_logits = out.logits[:, -1, :]
                else:
                    # Prompt IS the prefix; reuse last logits by running a dummy step
                    out = self.model(input_ids[:, -1:], past_key_values=past_kv, use_cache=True)
                    past_kv = out.past_key_values
                    last_logits = out.logits[:, -1, :]
                prefix_hit = True

        if past_kv is None:
            # Standard pre-fill: one shot over the full prompt
            past_kv, last_logits = self._prefill(input_ids)

        ttft = time.perf_counter() - t0
        generated_ids = []

        # Decode: pass only the single most-recent token each step
        for _ in range(max_new_tokens):
            if greedy:
                next_token = last_logits.argmax(dim=-1, keepdim=True)  # (1, 1)
            else:
                probs = torch.softmax(last_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)

            generated_ids.append(next_token.item())
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            out = self.model(next_token, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            last_logits = out.logits[:, -1, :]

        total = time.perf_counter() - t0
        full_ids = torch.cat(
            [input_ids, torch.tensor([generated_ids], device=self.device)], dim=-1
        )
        text = self.tokenizer.decode(full_ids[0], skip_special_tokens=True)

        return text, {
            "ttft_ms": ttft * 1000,
            "latency_ms": total * 1000,
            "new_tokens": len(generated_ids),
            "throughput_tps": len(generated_ids) / total,
            "prefix_hit": prefix_hit,
        }
