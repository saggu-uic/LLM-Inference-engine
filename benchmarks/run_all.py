"""
Run all benchmarks and print a comparison table.

Usage:
    python -m benchmarks.run_all                 # quick mode (default)
    python -m benchmarks.run_all --full          # more tokens, more repeats
    python -m benchmarks.run_all --skip-spec     # skip speculative (saves loading gpt2-medium)
"""
import argparse
import time
import torch

from benchmarks.metrics import BenchmarkResult, print_table
from engine.baseline import BaselineEngine
from engine.kv_cache import KVCacheEngine
from engine.continuous_batching import StaticBatchingEngine, ContinuousBatchingEngine
from engine.speculative import SpeculativeEngine
from engine.paged_attention import PagedAttentionEngine

# ─────────────────────────────────────────────────────────────────────────────
PROMPT = (
    "The key insight behind modern large language model serving is that "
    "attention computation can be cached across decode steps. Explain in detail"
)

BATCH_PROMPTS = [
    "Tell me a very short story about a cat.",
    "What is the capital of France?",
    "Write a haiku about autumn leaves.",
    "Explain quantum entanglement briefly.",
    "Describe the water cycle in one sentence.",
    "What are the primary colors?",
    "Write a limerick about a programmer.",
    "How does photosynthesis work?",
]
# Heterogeneous lengths: maximises the scheduling advantage for continuous batching
BATCH_MAX_TOKENS = [60, 15, 20, 40, 20, 10, 50, 35]


def avg(results: list[dict]) -> dict:
    keys = results[0].keys()
    return {k: sum(r[k] for r in results) / len(results) for k in keys}


def mem_mb() -> float:
    return torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0


def bench_baseline(n_tokens: int, n_repeats: int) -> BenchmarkResult:
    print("\n[1/6] Baseline (no KV cache) …")
    eng = BaselineEngine()
    stats = [eng.generate(PROMPT, max_new_tokens=n_tokens)[1] for _ in range(n_repeats)]
    s = avg(stats)
    return BenchmarkResult("Baseline (no KV cache)", s["ttft_ms"], s["throughput_tps"], s["latency_ms"], mem_mb())


def bench_kv_cache(n_tokens: int, n_repeats: int) -> list[BenchmarkResult]:
    print("\n[2/6] KV cache …")
    eng = KVCacheEngine()
    results = []

    stats = [eng.generate(PROMPT, max_new_tokens=n_tokens)[1] for _ in range(n_repeats)]
    s = avg(stats)
    results.append(BenchmarkResult("KV cache", s["ttft_ms"], s["throughput_tps"], s["latency_ms"], mem_mb()))

    SHARED = "The key insight behind modern large language model serving is that "
    SUFFIX = "attention computation can be cached across decode steps. Explain in detail"
    eng.cache_prefix(SHARED)
    stats2 = [eng.generate(SHARED + SUFFIX, max_new_tokens=n_tokens, shared_prefix=SHARED)[1]
              for _ in range(n_repeats)]
    s2 = avg(stats2)
    results.append(BenchmarkResult(
        "KV cache + prefix reuse", s2["ttft_ms"], s2["throughput_tps"], s2["latency_ms"], mem_mb(),
        extra={"prefix_hit": "yes"},
    ))
    return results


def bench_batching() -> list[BenchmarkResult]:
    print("\n[3/6] Continuous vs static batching …")
    results = []

    static_eng = StaticBatchingEngine()
    t0 = time.perf_counter()
    static_out = static_eng.run(BATCH_PROMPTS, BATCH_MAX_TOKENS, batch_size=4)
    time.perf_counter() - t0
    results.append(BenchmarkResult(
        "Static batching (B=4)", 0, static_out["throughput_tps"],
        static_out["total_time_s"] * 1000, mem_mb(),
        extra={"total_tok": static_out["total_tokens"]},
    ))

    cont_eng = ContinuousBatchingEngine()
    cont_out = cont_eng.run(BATCH_PROMPTS, BATCH_MAX_TOKENS, max_batch_size=4)
    results.append(BenchmarkResult(
        "Continuous batching (max=4)", 0, cont_out["throughput_tps"],
        cont_out["total_time_s"] * 1000, mem_mb(),
        extra={"total_tok": cont_out["total_tokens"]},
    ))
    return results


def bench_paged(n_tokens: int, n_repeats: int) -> BenchmarkResult:
    print("\n[4/6] Paged attention …")
    eng = PagedAttentionEngine()
    stats = [eng.generate(PROMPT, max_new_tokens=n_tokens)[1] for _ in range(n_repeats)]
    s = avg(stats)
    return BenchmarkResult(
        "Paged attention", s["ttft_ms"], s["throughput_tps"], s["latency_ms"], mem_mb(),
        extra={
            "pool_util": stats[-1]["pool_utilization"],
            "naive_kv": stats[-1]["naive_kv_mb"],
            "paged_kv": stats[-1]["paged_kv_mb"],
        },
    )


def bench_target_only(n_tokens: int, n_repeats: int) -> BenchmarkResult:
    print("\n[5/6] Target-only baseline (gpt2-medium + KV cache) …")
    eng = KVCacheEngine(model_name="gpt2-medium")
    stats = [eng.generate(PROMPT, max_new_tokens=n_tokens)[1] for _ in range(n_repeats)]
    s = avg(stats)
    return BenchmarkResult("gpt2-medium + KV cache", s["ttft_ms"], s["throughput_tps"], s["latency_ms"], mem_mb())


def bench_speculative(n_tokens: int, k: int = 4) -> BenchmarkResult:
    print(f"\n[6/6] Speculative decoding (gpt2 → gpt2-medium, k={k}) …")
    eng = SpeculativeEngine()
    _, stats = eng.generate(PROMPT, max_new_tokens=n_tokens, k=k)
    return BenchmarkResult(
        f"Speculative (k={k})", stats["ttft_ms"], stats["throughput_tps"], stats["latency_ms"], mem_mb(),
        extra={"accept": stats["accept_rate"], "tok/call": stats["tok_per_target_call"]},
    )


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="More tokens + repeats")
    parser.add_argument("--skip-spec", action="store_true", help="Skip speculative decoding")
    args = parser.parse_args()

    n_tokens = 80 if args.full else 40
    n_repeats = 3 if args.full else 1

    print(f"\n{'='*60}")
    print(f"  LLM Inference Benchmark  (tokens={n_tokens}, repeats={n_repeats})")
    print(f"  Device: {'CUDA ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*60}")

    all_results: list[BenchmarkResult] = []
    all_results.append(bench_baseline(n_tokens, n_repeats))
    all_results.extend(bench_kv_cache(n_tokens, n_repeats))
    all_results.extend(bench_batching())
    all_results.append(bench_paged(n_tokens, n_repeats))

    if not args.skip_spec:
        all_results.append(bench_target_only(n_tokens, n_repeats))
        all_results.append(bench_speculative(n_tokens))

    print_table(all_results)


if __name__ == "__main__":
    main()
