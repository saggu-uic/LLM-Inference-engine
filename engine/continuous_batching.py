"""
Continuous batching vs static batching.

Static batching:
  Group N requests into fixed batches of size B.  Pad all sequences to the
  LONGEST one in the batch.  The entire batch waits until every member
  finishes — short sequences burn GPU cycles computing attention over padding.

Continuous batching (iteration-level scheduling):
  Maintain a pool of in-flight sequences (up to max_batch_size).
  At every decode step, advance ALL in-flight sequences by one token.
  The moment a sequence reaches EOS or max_tokens, it leaves the pool and
  the next queued request takes its slot — no waiting for the rest of the batch.

The benchmark simulates a heterogeneous workload (mix of short and long outputs)
and measures total wall-clock time and aggregate throughput for both strategies.
"""
import time
import torch
from dataclasses import dataclass, field
from collections import deque
from transformers import GPT2LMHeadModel, GPT2Tokenizer


@dataclass
class _Sequence:
    request_id: int
    input_ids: torch.Tensor          # (1, prompt_len)
    past_kv: tuple | None
    last_logits: torch.Tensor | None
    max_new_tokens: int
    generated: list[int] = field(default_factory=list)
    t_start: float = field(default_factory=time.perf_counter)
    ttft: float | None = None


def _make_request(req_id, prompt_ids, max_new_tokens):
    return _Sequence(
        request_id=req_id,
        input_ids=prompt_ids,
        past_kv=None,
        last_logits=None,
        max_new_tokens=max_new_tokens,
    )


class StaticBatchingEngine:
    """
    Processes requests in fixed groups.  Pads each group to the longest prompt,
    then waits for ALL sequences in the group to emit EOS before moving on.
    """

    def __init__(self, model_name: str = "gpt2", device: str | None = None):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name, dtype=torch.float32)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device).eval()
        self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def run(
        self,
        prompts: list[str],
        max_new_tokens_per_request: list[int],
        batch_size: int = 4,
    ) -> dict:
        eos = self.tokenizer.eos_token_id
        t0 = time.perf_counter()
        total_generated = 0
        all_texts = []

        for batch_start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_start : batch_start + batch_size]
            batch_max = max_new_tokens_per_request[batch_start : batch_start + batch_size]

            enc = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            generated = input_ids.clone()
            # Track which sequences are still alive
            alive = torch.ones(len(batch_prompts), dtype=torch.bool)
            step_max = max(batch_max)

            for step in range(step_max):
                outputs = self.model(generated, attention_mask=attention_mask)
                logits = outputs.logits[:, -1, :]
                next_tokens = logits.argmax(dim=-1)  # (B,)

                # Sequences that have already finished keep emitting EOS (ignored)
                for i in range(len(batch_prompts)):
                    if step >= batch_max[i]:
                        alive[i] = False
                    if next_tokens[i].item() == eos:
                        alive[i] = False

                generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=-1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones(len(batch_prompts), 1, device=self.device)],
                    dim=-1,
                )

                if not alive.any():
                    break

            prompt_lens = enc["attention_mask"].sum(dim=-1)
            for i, (ids, plen) in enumerate(zip(generated, prompt_lens)):
                new = ids[plen:].tolist()
                if eos in new:
                    new = new[: new.index(eos)]
                total_generated += len(new)
                all_texts.append(
                    self.tokenizer.decode(ids[:plen].tolist() + new, skip_special_tokens=True)
                )

        total = time.perf_counter() - t0
        return {
            "strategy": "static",
            "total_time_s": total,
            "total_tokens": total_generated,
            "throughput_tps": total_generated / total,
            "texts": all_texts,
        }


class ContinuousBatchingEngine:
    """
    Iteration-level scheduling: finished sequences leave the pool immediately
    and new requests fill their slots without waiting for a full batch boundary.
    """

    def __init__(self, model_name: str = "gpt2", device: str | None = None):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name, dtype=torch.float32)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device).eval()

    @torch.no_grad()
    def _prefill_one(self, seq: _Sequence):
        """Run pre-fill for a single sequence; populate past_kv and last_logits."""
        out = self.model(seq.input_ids, use_cache=True)
        seq.past_kv = out.past_key_values
        seq.last_logits = out.logits[:, -1, :]
        seq.ttft = time.perf_counter() - seq.t_start

    @torch.no_grad()
    def _decode_one(self, seq: _Sequence) -> int:
        """Advance one decode step; return the new token id."""
        next_token = seq.last_logits.argmax(dim=-1, keepdim=True)  # greedy
        seq.generated.append(next_token.item())
        out = self.model(next_token, past_key_values=seq.past_kv, use_cache=True)
        seq.past_kv = out.past_key_values
        seq.last_logits = out.logits[:, -1, :]
        return next_token.item()

    def run(
        self,
        prompts: list[str],
        max_new_tokens_per_request: list[int],
        max_batch_size: int = 4,
    ) -> dict:
        eos = self.tokenizer.eos_token_id
        t0 = time.perf_counter()
        total_generated = 0
        all_texts = []
        finished: list[_Sequence] = []

        # Build request queue
        queue: deque[_Sequence] = deque()
        for i, (p, m) in enumerate(zip(prompts, max_new_tokens_per_request)):
            ids = self.tokenizer.encode(p, return_tensors="pt").to(self.device)
            queue.append(_make_request(i, ids, m))

        pool: list[_Sequence] = []

        while queue or pool:
            # Fill pool up to max_batch_size — prefill new arrivals immediately
            while queue and len(pool) < max_batch_size:
                seq = queue.popleft()
                self._prefill_one(seq)
                pool.append(seq)

            # One decode step for every in-flight sequence
            done_indices = []
            for idx, seq in enumerate(pool):
                tok = self._decode_one(seq)
                if tok == eos or len(seq.generated) >= seq.max_new_tokens:
                    done_indices.append(idx)

            # Remove finished sequences (back to front to preserve indices)
            for idx in reversed(done_indices):
                seq = pool.pop(idx)
                finished.append(seq)
                total_generated += len(seq.generated)
                full = seq.input_ids[0].tolist() + seq.generated
                all_texts.append(self.tokenizer.decode(full, skip_special_tokens=True))

        total = time.perf_counter() - t0
        # Reorder outputs to match input order
        finished.sort(key=lambda s: s.request_id)
        all_texts = [self.tokenizer.decode(
            s.input_ids[0].tolist() + s.generated, skip_special_tokens=True
        ) for s in finished]

        return {
            "strategy": "continuous",
            "total_time_s": total,
            "total_tokens": total_generated,
            "throughput_tps": total_generated / total,
            "texts": all_texts,
        }
