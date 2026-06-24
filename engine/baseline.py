"""
Baseline inference: naive autoregressive decoding — NO KV cache.

Every decode step re-runs attention over the FULL sequence from scratch.
Cost per step grows linearly with sequence length → O(n²) total compute.
This is the floor we beat with every other optimization.
"""
import time
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class BaselineEngine:
    def __init__(self, model_name: str = "gpt2", device: str | None = None):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name, dtype=torch.float32)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device).eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        greedy: bool = True,
        temperature: float = 1.0,
    ) -> tuple[str, dict]:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated = input_ids.clone()

        ttft = None
        t0 = time.perf_counter()

        for i in range(max_new_tokens):
            # Pass the FULL sequence every time — no past_key_values, no caching
            outputs = self.model(generated)
            logits = outputs.logits[:, -1, :]  # (1, vocab_size)

            if greedy:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)

            generated = torch.cat([generated, next_token], dim=-1)

            if ttft is None:
                ttft = time.perf_counter() - t0

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        total = time.perf_counter() - t0
        new_tokens = generated.shape[1] - input_ids.shape[1]
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)

        return text, {
            "ttft_ms": ttft * 1000,
            "latency_ms": total * 1000,
            "new_tokens": new_tokens,
            "throughput_tps": new_tokens / total,
        }
