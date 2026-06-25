"""
Paged Attention (Kwon et al., 2023 — vLLM).

Problem with standard KV cache
--------------------------------
Each sequence pre-allocates a CONTIGUOUS memory region for up to max_seq_len tokens.
Any request shorter than max_seq_len wastes the tail (internal fragmentation).
When many requests with different lengths share the GPU, total waste can exceed 60%.

Paged attention solution
-------------------------
Borrow the OS virtual-memory concept:

  1. Divide the KV cache into fixed-size BLOCKS (here: 16 tokens each).
  2. Each sequence has a BLOCK TABLE mapping logical block indices → physical block IDs.
  3. Allocate one block at a time, on demand. Free blocks immediately when a sequence ends.
  4. Multiple sequences that share a common prefix can point to the SAME physical blocks
     (copy-on-write prefix sharing — not implemented here but the allocator supports it).

Result: waste per sequence is at most (block_size - 1) tokens, independent of max_seq_len.

This implementation
--------------------
  - BlockAllocator  : manages a fixed pool of physical block IDs
  - PagedKVPool     : pre-allocated tensor (num_blocks × layers × 2 × heads × B × D)
  - PagedAttentionEngine : generates text, stores KV in the pool, gathers for attention

The gather step (assembling contiguous past_key_values before each model call) is the
main overhead vs production vLLM, which uses custom CUDA kernels that attend to
non-contiguous blocks directly without materialising a temporary tensor.  Our version
shows correct semantics and real memory accounting in pure PyTorch.
"""
from __future__ import annotations

import time
import math
from dataclasses import dataclass, field

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


BLOCK_SIZE = 16   # tokens per block  (vLLM default)


# ---------------------------------------------------------------------------
# Block allocator
# ---------------------------------------------------------------------------

class BlockAllocator:
    """
    Manages a pool of physical KV-cache blocks.

    Sequences grow into new blocks as they generate tokens; finished
    sequences release their blocks immediately so they can be reused.
    """

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self._free: list[int] = list(range(num_blocks))
        self._seq_blocks: dict[int, list[int]] = {}

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return self.num_blocks - len(self._free)

    @property
    def utilization(self) -> float:
        return self.num_used / self.num_blocks

    def alloc(self, seq_id: int) -> int:
        """Allocate one physical block for seq_id; return the block_id."""
        if not self._free:
            raise RuntimeError(
                f"OOM — all {self.num_blocks} blocks in use "
                f"({self.utilization:.0%} utilization)"
            )
        bid = self._free.pop()
        self._seq_blocks.setdefault(seq_id, []).append(bid)
        return bid

    def free(self, seq_id: int) -> None:
        """Release all blocks belonging to a finished sequence."""
        for bid in self._seq_blocks.pop(seq_id, []):
            self._free.append(bid)

    def block_table(self, seq_id: int) -> list[int]:
        """Return the list of physical block IDs for this sequence."""
        return list(self._seq_blocks.get(seq_id, []))


# ---------------------------------------------------------------------------
# Pre-allocated KV pool
# ---------------------------------------------------------------------------

class PagedKVPool:
    """
    Single pre-allocated tensor that holds ALL sequences' KV vectors.

    Layout: [num_blocks, num_layers, 2, num_heads, block_size, head_dim]
              index 0        1       2      3           4          5
    The '2' axis is K=0 / V=1.

    Slicing by block_id gives us the KV for one block across all layers.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
    ):
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        # Allocate once, reuse forever
        self.data = torch.zeros(
            num_blocks, num_layers, 2, num_heads, block_size, head_dim,
            device=device, dtype=torch.float32,
        )
        self.memory_bytes = self.data.nelement() * self.data.element_size()
        self.memory_mb = self.memory_bytes / 1024 ** 2

    def write(
        self,
        block_id: int,
        slot: int,       # position within the block (0 … block_size-1)
        layer: int,
        k: torch.Tensor, # (num_heads, head_dim)
        v: torch.Tensor,
    ) -> None:
        self.data[block_id, layer, 0, :, slot, :] = k
        self.data[block_id, layer, 1, :, slot, :] = v

    def gather(
        self,
        block_ids: list[int],
        seq_len: int,
        layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Assemble K, V for all seq_len tokens by following the block table.

        Returns K, V each shaped (1, num_heads, seq_len, head_dim).

        In production paged attention (vLLM) a custom CUDA kernel reads the
        non-contiguous blocks directly — no gather needed.  We cat here to keep
        the implementation in pure PyTorch.
        """
        k_parts: list[torch.Tensor] = []
        v_parts: list[torch.Tensor] = []
        remaining = seq_len

        for bid in block_ids:
            slots = min(self.block_size, remaining)
            # self.data[bid, layer, 0] shape: (num_heads, block_size, head_dim)
            k_parts.append(self.data[bid, layer, 0, :, :slots, :])
            v_parts.append(self.data[bid, layer, 1, :, :slots, :])
            remaining -= slots
            if remaining == 0:
                break

        k = torch.cat(k_parts, dim=1).unsqueeze(0)  # (1, H, seq_len, D)
        v = torch.cat(v_parts, dim=1).unsqueeze(0)
        return k, v


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class PagedAttentionEngine:
    """
    KV-cache inference backed by paged memory management.

    Generates the same text as KVCacheEngine.  The difference is in memory:
    instead of growing a per-sequence contiguous tensor, KV vectors are written
    into fixed-size blocks from a shared pool that is allocated up-front once.

    Key metrics logged per generation:
      pool_utilization  — fraction of blocks in use during the run
      naive_kv_mb       — memory a naive (contiguous) cache would have used
      paged_kv_mb       — memory our paged scheme actually uses
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        num_blocks: int = 64,
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
        self.num_heads: int = cfg.n_head
        self.head_dim: int = cfg.n_embd // cfg.n_head

        self.allocator = BlockAllocator(num_blocks)
        self.pool = PagedKVPool(
            num_blocks, block_size,
            self.num_layers, self.num_heads, self.head_dim,
            self.device,
        )
        print(
            f"  KV pool: {num_blocks} blocks × {block_size} tok/block "
            f"= {num_blocks * block_size} slots  "
            f"({self.pool.memory_mb:.1f} MB pre-allocated)"
        )

    # ------------------------------------------------------------------ helpers

    def _blocks_needed(self, seq_len: int) -> int:
        return math.ceil(seq_len / self.block_size)

    def _ensure_blocks(self, seq_id: int, old_len: int, new_len: int) -> None:
        """Allocate new blocks if the sequence has grown into a new page."""
        old_b = self._blocks_needed(old_len) if old_len > 0 else 0
        new_b = self._blocks_needed(new_len)
        for _ in range(new_b - old_b):
            self.allocator.alloc(seq_id)

    def _store_kv(
        self,
        seq_id: int,
        past_key_values: tuple,
        start_pos: int,
        end_pos: int,
    ) -> None:
        """Write KV vectors for positions [start_pos, end_pos) into the pool."""
        block_table = self.allocator.block_table(seq_id)
        for pos in range(start_pos, end_pos):
            block_idx = pos // self.block_size
            slot = pos % self.block_size
            phys_block = block_table[block_idx]
            for layer in range(self.num_layers):
                k = past_key_values[layer][0][0, :, pos, :]  # (H, D)
                v = past_key_values[layer][1][0, :, pos, :]
                self.pool.write(phys_block, slot, layer, k, v)

    def _build_past_kv(self, seq_id: int, seq_len: int) -> tuple:
        """Reconstruct past_key_values by gathering from the paged pool."""
        block_table = self.allocator.block_table(seq_id)
        return tuple(
            self.pool.gather(block_table, seq_len, layer)
            for layer in range(self.num_layers)
        )

    # ------------------------------------------------------------------ generate

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        greedy: bool = True,
        seq_id: int = 0,
    ) -> tuple[str, dict]:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_len = input_ids.shape[1]
        t0 = time.perf_counter()

        # ---- pre-fill ----
        self._ensure_blocks(seq_id, 0, prompt_len)
        out = self.model(input_ids, use_cache=True)
        self._store_kv(seq_id, out.past_key_values, 0, prompt_len)
        last_logits = out.logits[:, -1, :]
        ttft = time.perf_counter() - t0

        generated_ids: list[int] = []
        seq_len = prompt_len

        # ---- decode ----
        for _ in range(max_new_tokens):
            next_tok = (
                last_logits.argmax(dim=-1, keepdim=True) if greedy
                else torch.multinomial(torch.softmax(last_logits, dim=-1), 1)
            )
            tok_id = next_tok.item()
            generated_ids.append(tok_id)
            if tok_id == self.tokenizer.eos_token_id:
                break

            # Extend page table if this token starts a new block
            self._ensure_blocks(seq_id, seq_len, seq_len + 1)
            seq_len += 1

            # Gather past KV from pages → run model → store new token's KV
            past = self._build_past_kv(seq_id, seq_len - 1)
            out = self.model(next_tok, past_key_values=past, use_cache=True)
            self._store_kv(seq_id, out.past_key_values, seq_len - 1, seq_len)
            last_logits = out.logits[:, -1, :]

        # ---- cleanup ----
        peak_utilization = self.allocator.utilization
        blocks_used = self.allocator.num_used
        self.allocator.free(seq_id)

        total = time.perf_counter() - t0
        new_tokens = len(generated_ids)
        full_ids = torch.cat(
            [input_ids, torch.tensor([generated_ids], device=self.device)], dim=-1
        )
        text = self.tokenizer.decode(full_ids[0], skip_special_tokens=True)

        # Memory comparison: how much a naive contiguous cache would have used
        per_tok = self.num_layers * 2 * self.num_heads * self.head_dim * 4  # bytes
        total_tokens = prompt_len + new_tokens
        naive_mb = total_tokens * per_tok / 1024 ** 2
        paged_mb = self._blocks_needed(total_tokens) * self.block_size * per_tok / 1024 ** 2

        return text, {
            "ttft_ms": ttft * 1000,
            "latency_ms": total * 1000,
            "new_tokens": new_tokens,
            "throughput_tps": new_tokens / total,
            "pool_utilization": f"{peak_utilization:.0%}",
            "blocks_used": blocks_used,
            "naive_kv_mb": f"{naive_mb:.3f} MB",
            "paged_kv_mb": f"{paged_mb:.3f} MB",
        }

    # ------------------------------------------------------------------ memory demo

    def memory_comparison(
        self,
        prompts: list[str],
        max_new_tokens_list: list[int],
    ) -> dict:
        """
        Pure memory accounting — no inference.

        Computes how much KV memory a naive (contiguous, max_len-pre-allocated)
        cache would need vs the paged approach for a heterogeneous batch.

        Naive pre-allocates max(max_new_tokens) for every request regardless
        of how many tokens each actually generates.  Paged allocates blocks
        on demand, so waste per request is at most block_size-1 tokens.
        """
        per_tok = self.num_layers * 2 * self.num_heads * self.head_dim * 4  # bytes
        max_len = max(max_new_tokens_list)

        prompt_lens = [len(self.tokenizer.encode(p)) for p in prompts]
        actual_lens = [pl + mt for pl, mt in zip(prompt_lens, max_new_tokens_list)]

        # Naive: every sequence gets a contiguous buffer of (prompt + max_len) tokens
        naive_bytes = sum((pl + max_len) * per_tok for pl in prompt_lens)

        # Paged: each sequence only uses ceil(actual_len / block_size) blocks
        paged_bytes = sum(
            self._blocks_needed(al) * self.block_size * per_tok
            for al in actual_lens
        )

        return {
            "num_sequences": len(prompts),
            "max_new_tokens": max_len,
            "actual_new_tokens": max_new_tokens_list,
            "naive_mb": naive_bytes / 1024 ** 2,
            "paged_mb": paged_bytes / 1024 ** 2,
            "savings_mb": (naive_bytes - paged_bytes) / 1024 ** 2,
            "savings_pct": (1 - paged_bytes / naive_bytes) * 100,
        }
