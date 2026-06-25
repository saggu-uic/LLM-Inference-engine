"""
Paged Continuous Batching — combines paged KV memory management
with iteration-level scheduling (the architecture used by vLLM).

Why each technique alone is insufficient
-----------------------------------------
Continuous batching (engine/continuous_batching.py):
  Great throughput — GPU is never idle waiting for laggards.
  Bad memory — each sequence holds a DynamicCache that grows without
  bound and occupies CONTIGUOUS VRAM.  With many in-flight sequences,
  fragmentation quickly exhausts the pool.

Paged attention (engine/paged_attention.py):
  Great memory — KV vectors live in fixed-size shared blocks, freed
  the instant a sequence ends.
  No inherent batching — the single-sequence loop leaves GPU
  underutilised between requests.

Combined: Paged Continuous Batching
--------------------------------------
  1. Iteration-level scheduler: advance ALL in-flight sequences one
     token per step; slot is freed the moment a sequence hits EOS.
  2. Shared block pool: each sequence's KV lives in the pool, not in
     a per-sequence DynamicCache that inflates VRAM.

Effect:
  - Throughput ≈ continuous batching (GPU always busy)
  - Memory    ≈ paged attention (blocks freed immediately, no fragmentation)

In vLLM this is combined with prefix caching, chunked prefill, and
speculative decoding.  Here we show the core mechanism in pure PyTorch.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.cache_utils import DynamicCache

from .paged_attention import BlockAllocator, PagedKVPool, BLOCK_SIZE


# ---------------------------------------------------------------------------
# Internal sequence state
# ---------------------------------------------------------------------------

@dataclass
class _Seq:
    request_id: int
    seq_id: int                           # key in BlockAllocator
    input_ids: list[int]
    max_new_tokens: int
    seq_len: int = 0                      # prompt_len + tokens generated so far
    last_logits: torch.Tensor | None = None
    generated: list[int] = field(default_factory=list)
    t_start: float = field(default_factory=time.perf_counter)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PagedContinuousBatchingEngine:
    """
    Iteration-level scheduler backed by a paged KV block pool.

    API mirrors ContinuousBatchingEngine.run() so the two can be compared
    directly in benchmarks.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        num_blocks: int = 128,
        block_size: int = BLOCK_SIZE,
        device: str | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.block_size = block_size

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name, dtype=torch.float32)
        self.model.to(self.device).eval()

        cfg = self.model.config
        self.num_layers: int = cfg.n_layer
        self.num_heads: int  = cfg.n_head
        self.head_dim: int   = cfg.n_embd // cfg.n_head

        self.allocator = BlockAllocator(num_blocks)
        self.kv_pool = PagedKVPool(
            num_blocks, block_size,
            self.num_layers, self.num_heads, self.head_dim,
            self.device,
        )
        print(
            f"  KV pool: {num_blocks} blocks × {block_size} tok/block "
            f"= {num_blocks * block_size} slots  "
            f"({self.kv_pool.memory_mb:.1f} MB pre-allocated)"
        )

    # ------------------------------------------------------------------ paged KV helpers
    # (same DynamicCache-agnostic API as PagedAttentionEngine)

    def _sync(self):
        """Wait for queued CUDA kernels so wall-clock timing is real. No-op on CPU."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _blocks_needed(self, length: int) -> int:
        return math.ceil(length / self.block_size)

    def _ensure_blocks(self, seq_id: int, old_len: int, new_len: int) -> None:
        old_b = self._blocks_needed(old_len) if old_len > 0 else 0
        new_b = self._blocks_needed(new_len)
        for _ in range(new_b - old_b):
            self.allocator.alloc(seq_id)

    @staticmethod
    def _to_legacy(cache) -> tuple:
        """
        Extract per-layer (k, v) tensors from a DynamicCache, across versions.
          * >= 4.54 : cache.layers[i].keys / .values   (current — verified 4.57.x)
          * 4.36-4.53: cache.key_cache[i] / .value_cache[i]
          * deprecated fallback: to_legacy_cache()
        """
        layers = getattr(cache, "layers", None)
        if layers is not None:
            return tuple((lyr.keys, lyr.values) for lyr in layers)
        if hasattr(cache, "key_cache"):
            return tuple(
                (cache.key_cache[i], cache.value_cache[i])
                for i in range(len(cache.key_cache))
            )
        if hasattr(cache, "to_legacy_cache"):
            return cache.to_legacy_cache()
        raise RuntimeError(f"Unsupported KV cache layout: {type(cache)}")

    def _store_kv(self, seq_id: int, cache, start: int, end: int) -> None:
        block_table = self.allocator.block_table(seq_id)
        kv_pairs = self._to_legacy(cache)
        for pos in range(start, end):
            phys = block_table[pos // self.block_size]
            slot = pos % self.block_size
            for layer, (k, v) in enumerate(kv_pairs):
                self.kv_pool.write(phys, slot, layer, k[0, :, pos, :], v[0, :, pos, :])

    def _build_cache(self, seq_id: int, seq_len: int) -> DynamicCache:
        """Rebuild a DynamicCache from paged blocks via update() — the one
        Cache API stable across every transformers version."""
        block_table = self.allocator.block_table(seq_id)
        cache = DynamicCache()
        for layer in range(self.num_layers):
            k, v = self.kv_pool.gather(block_table, seq_len, layer)
            cache.update(k, v, layer)
        return cache

    # ------------------------------------------------------------------ sequence lifecycle

    @torch.no_grad()
    def _prefill(self, seq: _Seq) -> None:
        """Run full prompt; store KV in block pool."""
        ids = torch.tensor([seq.input_ids], device=self.device)
        prompt_len = len(seq.input_ids)
        self._ensure_blocks(seq.seq_id, 0, prompt_len)
        out = self.model(ids, use_cache=True)
        self._store_kv(seq.seq_id, out.past_key_values, 0, prompt_len)
        seq.last_logits = out.logits[:, -1, :]
        seq.seq_len = prompt_len

    @torch.no_grad()
    def _decode(self, seq: _Seq) -> int:
        """
        One decode step: generate next token, extend block table if
        needed, store new token's KV back into the pool.
        """
        next_tok = seq.last_logits.argmax(dim=-1, keepdim=True)
        tok_id = next_tok.item()
        seq.generated.append(tok_id)

        self._ensure_blocks(seq.seq_id, seq.seq_len, seq.seq_len + 1)
        seq.seq_len += 1

        past = self._build_cache(seq.seq_id, seq.seq_len - 1)
        out = self.model(next_tok, past_key_values=past, use_cache=True)
        self._store_kv(seq.seq_id, out.past_key_values, seq.seq_len - 1, seq.seq_len)
        seq.last_logits = out.logits[:, -1, :]
        return tok_id

    # ------------------------------------------------------------------ public API

    def run(
        self,
        prompts: list[str],
        max_new_tokens_per_request: list[int],
        max_batch_size: int = 4,
    ) -> dict:
        """
        Process all prompts with paged-continuous scheduling.

        Returns the same keys as ContinuousBatchingEngine.run() PLUS
        memory accounting keys (peak_pool_util, naive_kv_mb, paged_kv_mb).
        """
        eos = self.tokenizer.eos_token_id
        self._sync()
        t0 = time.perf_counter()
        finished: list[_Seq] = []
        peak_util = 0.0

        queue: deque[_Seq] = deque()
        for i, (p, m) in enumerate(zip(prompts, max_new_tokens_per_request)):
            ids = self.tokenizer.encode(p)
            queue.append(_Seq(request_id=i, seq_id=i, input_ids=ids, max_new_tokens=m))

        pool: list[_Seq] = []

        while queue or pool:
            # Fill empty slots — prefill new arrivals immediately
            while queue and len(pool) < max_batch_size:
                seq = queue.popleft()
                self._prefill(seq)
                pool.append(seq)

            peak_util = max(peak_util, self.allocator.utilization)

            # Advance every in-flight sequence by one token
            done: list[int] = []
            for idx, seq in enumerate(pool):
                tok = self._decode(seq)
                if tok == eos or len(seq.generated) >= seq.max_new_tokens:
                    done.append(idx)

            # Remove finished sequences and return their blocks immediately
            # (back-to-front to keep indices stable)
            for idx in reversed(done):
                seq = pool.pop(idx)
                self.allocator.free(seq.seq_id)   # ← blocks back in free list now
                finished.append(seq)

        self._sync()
        total = time.perf_counter() - t0
        total_generated = sum(len(s.generated) for s in finished)

        # Memory comparison
        per_tok = self.num_layers * 2 * self.num_heads * self.head_dim * 4  # bytes
        max_gen = max(max_new_tokens_per_request)
        finished.sort(key=lambda s: s.request_id)

        # Naive: each request pre-allocates a contiguous (prompt + max_gen) buffer
        naive_bytes = sum((len(s.input_ids) + max_gen) * per_tok for s in finished)
        # Paged: only the blocks actually used per sequence
        paged_bytes = sum(
            self._blocks_needed(len(s.input_ids) + len(s.generated)) * self.block_size * per_tok
            for s in finished
        )

        texts = [
            self.tokenizer.decode(s.input_ids + s.generated, skip_special_tokens=True)
            for s in finished
        ]

        return {
            "strategy": "paged_continuous",
            "total_time_s": total,
            "total_tokens": total_generated,
            "throughput_tps": total_generated / total,
            "peak_pool_util": f"{peak_util:.0%}",
            "naive_kv_mb": f"{naive_bytes / 1024 ** 2:.2f} MB",
            "paged_kv_mb": f"{paged_bytes / 1024 ** 2:.2f} MB",
            "savings_pct": f"{(1 - paged_bytes / naive_bytes) * 100:.1f}%",
            "texts": texts,
        }
