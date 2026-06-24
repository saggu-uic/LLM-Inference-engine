"""
Run all five benchmarks and print a comparison table.

Usage:
    python -m benchmarks.run_all                 # quick mode (default)
    python -m benchmarks.run_all --full          # more tokens, more repeats
    python -m benchmarks.run_all --skip-spec     # skip speculative (saves loading gpt2-medium)
"""
import argparse
import sys
import time
import torch

from benchmarks.metrics import BenchmarkResult, print_table
from engine.baseline import BaselineEngine
from engine.kv_cache import KVCacheEngine
from engine.continuous_batching import StaticBatchingEngine, ContinuousBatchingEngine
from engine.speculative import SpeculativeEngine
from engine.quantization import QuantizationEngine

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
# Mix short and long outputs to stress-test batching strategies
BATCH_MAX_TOKENS = [60, 15, 20, 40, 20, 10, 50, 35]


def avg(results: list[dict]) -> dict:
    keys = results[0].keys()
    return {k: sum(r[k] for r in results) / len(results) for k in keys}


def bench_baseline(n_tokens: int, n_repeats: int) -> BenchmarkResult:
    print("\n[1/6] Baseline (no KV cache) …")
    eng = BaselineEngine()
    stats = [eng.generate(PROMPT, max_new_tokens=n_tokens)[1] for _ in range(n_repeats)]
    s = avg(stats)
    return BenchmarkResult(
        name="Baseline (no KV cache)",
        ttft_ms=s["ttft_ms"],
        throughput_tps=s["throughput_tps"],
        latency_ms=s["latency_ms"],
        memory_mb=torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0,
    )


def bench_kv_cache(n_tokens: int, n_repeats: int) -> list[BenchmarkResult]:
    print("\n[2/6] KV cache …")
    eng = KVCacheEngine()
    results = []

    # Without prefix reuse
    stats = [eng.generate(PROMPT, max_new_tokens=n_tokens)[1] for _ in range(n_repeats)]
    s = avg(stats)
    results.append(BenchmarkResult(
        name="KV cache",
        ttft_ms=s["ttft_ms"],
        throughput_tps=s["throughput_tps"],
        latency_ms=s["latency_ms"],
        memory_mb=torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0,
    ))

    # With prefix reuse — cache the shared prefix, then run a shorter suffix
    SHARED = "The key insight behind modern large language model serving is that "
    SUFFIX = "attention computation can be cached across decode steps. Explain in detail"
    eng.cache_prefix(SHARED)
    full_prompt = SHARED + SUFFIX
    stats2 = [eng.generate(full_prompt, max_new_tokens=n_tokens, shared_prefix=SHARED)[1]
              for _ in range(n_repeats)]
    s2 = avg(stats2)
    results.append(BenchmarkResult(
        name="KV cache + prefix reuse",
        ttft_ms=s2["ttft_ms"],
        throughput_tps=s2["throughput_tps"],
        latency_ms=s2["latency_ms"],
        memory_mb=torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0,
        extra={"prefix_hit": "yes"},
    ))
    return results


def bench_batching() -> list[BenchmarkResult]:
    print("\n[3/6] Continuous vs static batching …")
    results = []

    static_eng = StaticBatchingEngine()
    t0 = time.perf_counter()
    static_out = static_eng.run(BATCH_PROMPTS, BATCH_MAX_TOKENS, batch_size=4)
    static_wall = time.perf_counter() - t0

    results.append(BenchmarkResult(
        name="Static batching (B=4)",
        ttft_ms=0,   # not tracked per-request in static mode
        throughput_tps=static_out["throughput_tps"],
        latency_ms=static_out["total_time_s"] * 1000,
        memory_mb=torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0,
        extra={"total_tok": static_out["total_tokens"]},
    ))

    cont_eng = ContinuousBatchingEngine()
    cont_out = cont_eng.run(BATCH_PROMPTS, BATCH_MAX_TOKENS, max_batch_size=4)

    results.append(BenchmarkResult(
        name="Continuous batching (max=4)",
        ttft_ms=0,
        throughput_tps=cont_out["throughput_tps"],
        latency_ms=cont_out["total_time_s"] * 1000,
        memory_mb=torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0,
        extra={"total_tok": cont_out["total_tokens"]},
    ))
    return results


def bench_speculative(n_tokens: int, k: int = 4) -> BenchmarkResult:
    print("\n[4/6] Speculative decoding (gpt2 → gpt2-medium) …")
    eng = SpeculativeEngine()
    _, stats = eng.generate(PROMPT, max_new_tokens=n_tokens, k=k)
    return BenchmarkResult(
        name=f"Speculative (k={k})",
        ttft_ms=stats["ttft_ms"],
        throughput_tps=stats["throughput_tps"],
        latency_ms=stats["latency_ms"],
        memory_mb=torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0,
        extra={
            "accept": stats["accept_rate"],
            "tok/call": stats["tok_per_target_call"],
        },
    )


def bench_target_only(n_tokens: int, n_repeats: int) -> BenchmarkResult:
    """KV-cache inference with the target model (gpt2-medium) as a fair baseline."""
    print("\n[5/6] Target-only baseline (gpt2-medium + KV cache) …")
    eng = KVCacheEngine(model_name="gpt2-medium")
    stats = [eng.generate(PROMPT, max_new_tokens=n_tokens)[1] for _ in range(n_repeats)]
    s = avg(stats)
    return BenchmarkResult(
        name="gpt2-medium + KV cache",
        ttft_ms=s["ttft_ms"],
        throughput_tps=s["throughput_tps"],
        latency_ms=s["latency_ms"],
        memory_mb=torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0,
    )


def bench_quantization(n_tokens: int) -> list[BenchmarkResult]:
    print("\n[6/6] Quantization (FP32 → INT8 → W4) …")
    results = []
    for mode in ("fp32", "int8", "w4"):
        eng = QuantizationEngine(mode=mode)
        _, stats = eng.generate(PROMPT, max_new_tokens=n_tokens)
        results.append(BenchmarkResult(
            name=f"Quantization ({mode.upper()})",
            ttft_ms=stats["ttft_ms"],
            throughput_tps=stats["throughput_tps"],
            latency_ms=stats["latency_ms"],
            memory_mb=torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0,
            extra={"size": stats["model_size_mb"]},
        ))
    return results


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="More tokens + repeats")
    parser.add_argument("--skip-spec", action="store_true", help="Skip speculative decoding")
    parser.add_argument("--skip-quant", action="store_true", help="Skip quantization")
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

    if not args.skip_spec:
        all_results.append(bench_target_only(n_tokens, n_repeats))
        all_results.append(bench_speculative(n_tokens))

    if not args.skip_quant:
        all_results.extend(bench_quantization(n_tokens))

    print_table(all_results)


if __name__ == "__main__":
    main()
