# LLM Inference Engine from Scratch

A from-scratch implementation of the core optimizations that power modern LLM
serving stacks (vLLM, TGI), built on GPT-2 in pure PyTorch with before/after
benchmarks. Every technique is implemented as a self-contained engine so you can
read it, run it, and see the numbers.

**Goal:** understand *why* each optimization exists by building it and measuring it —
not by reading about it.

---

## Quickstart

### Colab (recommended — free T4 GPU)
Open [`colab_demo.ipynb`](colab_demo.ipynb) in Google Colab and run all cells.
The notebook clones the repo, installs deps, and walks through every engine with
explanations and live numbers.

### Local
```bash
pip install -r requirements.txt

python demo.py                 # one generation per engine (smoke test)
python -m benchmarks.run_all   # full benchmark table
python -m benchmarks.run_all --skip-spec   # skip speculative (avoids downloading gpt2-medium)
```
> Note: the speculative engine needs `gpt2-medium` (~1.5 GB). Everything else
> runs on `gpt2` (117 M).

---

## What we built

Six inference engines, each isolating one optimization:

| # | Engine | File | Optimizes |
|---|--------|------|-----------|
| 1 | Baseline (no cache) | [`engine/baseline.py`](engine/baseline.py) | — (reference floor) |
| 2 | KV cache + prefix reuse | [`engine/kv_cache.py`](engine/kv_cache.py) | Latency |
| 3 | Static vs continuous batching | [`engine/continuous_batching.py`](engine/continuous_batching.py) | Throughput / scheduling |
| 4 | Paged attention | [`engine/paged_attention.py`](engine/paged_attention.py) | Memory |
| 5 | Paged continuous batching | [`engine/paged_continuous.py`](engine/paged_continuous.py) | Memory + scheduling |
| 6 | Speculative decoding | [`engine/speculative.py`](engine/speculative.py) | Target-model calls |

---

## Architecture

```
Inference/
├── engine/                      # one optimization per file
│   ├── baseline.py              # naive autoregressive decoding, no KV cache
│   ├── kv_cache.py              # KV cache + shared-prefix reuse
│   ├── continuous_batching.py   # StaticBatchingEngine + ContinuousBatchingEngine
│   ├── paged_attention.py       # BlockAllocator + PagedKVPool + PagedAttentionEngine
│   ├── paged_continuous.py      # paged block pool + iteration-level scheduler
│   └── speculative.py           # draft→target speculative decoding (k=4)
├── benchmarks/
│   ├── metrics.py               # BenchmarkResult, Timer, MemoryTracker, print_table
│   └── run_all.py               # runs every engine, prints comparison table
├── demo.py                      # quick smoke test (one generation per engine)
└── colab_demo.ipynb             # annotated walkthrough for Colab T4
```

**Model:** GPT-2 (12 layers, 12 heads, head_dim 64). Speculative uses
`gpt2` (draft) → `gpt2-medium` (target).

### How the optimizations work

**KV cache** — Split inference into *prefill* (run the whole prompt once, store K/V
for every layer) and *decode* (each new token attends to stored K/V, O(1) extra
work per step instead of O(n)). **Prefix reuse** stores the KV of a shared system
prompt so repeated requests skip re-prefilling it.

**Continuous batching** — Static batching pads every request to the longest in the
batch and makes the whole batch wait for the slowest to finish. Continuous batching
schedules at *iteration* granularity: the moment a sequence finishes, its slot is
filled by the next queued request — no waiting on a batch boundary.

**Paged attention** (Kwon et al., 2023) — Instead of a contiguous per-sequence KV
buffer pre-sized to `max_len` (which wastes the unused tail), KV is stored in
fixed-size **blocks** drawn from a shared pool. A per-sequence **block table** maps
logical→physical blocks; blocks are allocated on demand and freed instantly when a
sequence ends. Waste per sequence drops to at most `block_size − 1` tokens,
independent of `max_len`.

**Paged continuous batching** — Combines the two above: the iteration-level
scheduler keeps the GPU busy while the shared block pool keeps memory flat. This is
the core architecture of vLLM.

**Speculative decoding** (Chen et al., 2023) — A small *draft* model proposes `k`
tokens cheaply; the large *target* model verifies all `k` in **one** forward pass.
Rejection sampling guarantees the output distribution is identical to sampling from
the target alone. KV is managed with `cache.crop()`: the verify pass advances the
target cache over all `k` positions, then rejected positions are cropped off — so
the target runs only **2 times per step** while producing `n_accepted + 1` tokens.

---

## Results

Measured on a **Google Colab T4 GPU**, 80 generated tokens, transformers 4.57.x.
GPU timings use `torch.cuda.synchronize()` so they reflect compute, not async
kernel-launch time.

| Engine | TTFT | Throughput | Notes |
|--------|------|-----------|-------|
| Baseline (no KV cache) | 596 ms | 47.5 tok/s | reference floor |
| **KV cache** | **14 ms** | 87.7 tok/s | **42× faster TTFT** |
| KV cache + prefix reuse | — | 82.4 tok/s | see caveat below |
| Static batching (B=4) | — | 135.1 tok/s | 250 tokens, one batched forward |
| Continuous batching (max=4) | — | 99.5 tok/s | 250 tokens, better scheduling |
| Paged continuous (max=4) | — | 87.9 tok/s | 250 tokens, **35% less KV memory** |
| Paged attention | 35 ms | 58.9 tok/s | **34.5% less KV memory** (8-request mix) |
| gpt2-medium (KV cache) | 21 ms | 52.4 tok/s | speculative baseline |
| **Speculative (k=4)** | 30 ms | 39.0 tok/s | **68.6% accept, 1.84 tokens/target-call** |

### The three headline wins

1. **KV cache → 42× faster time-to-first-token.** The single largest, cleanest result.
2. **Paged attention → 34.5% less KV memory** on a heterogeneous batch (and 35% in the
   paged-continuous engine). Memory waste becomes `block_size − 1` per sequence.
3. **Speculative decoding → 1.84 tokens per target-model call** (at 68.6% accept rate).
   The expensive target model runs ~1.84× *fewer* times than tokens produced — the
   metric that proves the mechanism works.

### Honest caveats (read before quoting numbers)

This is a **pure-PyTorch eager** demo on **small** models. That cleanly demonstrates
*latency* (KV cache) and *memory* (paged attention) wins, but **throughput** wins are
understated:

- **Prefix reuse TTFT looks slow on GPU** only because the demo's shared prefix is ~12
  tokens — too short to overcome fixed kernel overhead. The logic is correct (on CPU,
  reuse is measurably faster); the benefit shows with a long, real system prompt.
- **Continuous batching is slower than static here** because static does one *batched*
  forward (GPU-parallel) while this teaching implementation advances sequences one at a
  time in Python. Closing that gap requires batched paged-attention CUDA kernels — which
  is exactly the problem vLLM exists to solve.
- **Paged engines trade a little speed for memory.** The pure-PyTorch "gather" step that
  assembles contiguous KV before each forward is overhead that production vLLM avoids
  with a custom kernel. The win to read here is **memory**, not throughput.
- **Speculative wall-clock is ~target speed** on this model pair: running two models plus
  a Python acceptance loop in eager mode eats the savings. The win to read is
  **tokens/target-call**; wall-clock speedup needs a larger draft↔target gap and
  optimized kernels.

---

## Implementation notes

- **transformers 4.57.x `DynamicCache`** stores KV in `cache.layers[i].keys/.values`
  (the older `.key_cache` / `to_legacy_cache()` were removed). Engines extract KV via
  `cache.layers`, rebuild via `DynamicCache().update(...)`, and trim via `cache.crop(...)`
  — all version-robust public APIs.
- All timing is wrapped in `torch.cuda.synchronize()` (a no-op on CPU) so GPU numbers
  measure compute.
- Loading multiple models in a single process can segfault on some Windows setups
  (a native threading quirk, not a code bug) — `demo.py` loads engines sequentially.

## References

- Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (2023) — vLLM
- Chen et al., *Accelerating LLM Decoding with Speculative Sampling* (2023)
- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding* (2023)
