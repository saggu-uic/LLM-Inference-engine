"""
Quick smoke-test: runs one generation from each engine and prints stats.
No benchmarking, just sanity-checks that every module works.

Usage:  python demo.py
"""
import torch

PROMPT = "The meaning of life is"
MAX_TOKENS = 30


def divider(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


def show(label, text, stats):
    print(f"\n[{label}]")
    print(f"  Text  : {text[:120]}")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:<20}: {v:.2f}")
        else:
            print(f"  {k:<20}: {v}")


# ── 1. Baseline ──────────────────────────────────────────────────────────────
divider("1. Baseline (no KV cache)")
from engine.baseline import BaselineEngine
eng = BaselineEngine()
text, stats = eng.generate(PROMPT, max_new_tokens=MAX_TOKENS)
show("baseline", text, stats)

# ── 2. KV Cache ──────────────────────────────────────────────────────────────
divider("2. KV Cache")
from engine.kv_cache import KVCacheEngine
eng = KVCacheEngine()
text, stats = eng.generate(PROMPT, max_new_tokens=MAX_TOKENS)
show("kv_cache", text, stats)

# KV with prefix reuse
SHARED = "The meaning of "
eng.cache_prefix(SHARED)
text2, stats2 = eng.generate(SHARED + "life is", max_new_tokens=MAX_TOKENS, shared_prefix=SHARED)
show("kv_cache + prefix_reuse", text2, stats2)

# ── 3. Continuous batching ───────────────────────────────────────────────────
divider("3. Continuous Batching vs Static Batching")
from engine.continuous_batching import StaticBatchingEngine, ContinuousBatchingEngine

prompts = [
    "The sky is",
    "Once upon a time in a land far away there lived",
    "Python is",
    "A long time ago in a galaxy far far away",
]
max_toks = [10, 40, 10, 40]

static_eng = StaticBatchingEngine()
s_out = static_eng.run(prompts, max_toks, batch_size=4)
print(f"\n  Static  : {s_out['total_time_s']:.2f}s | {s_out['throughput_tps']:.1f} tok/s | {s_out['total_tokens']} tokens")

cont_eng = ContinuousBatchingEngine()
c_out = cont_eng.run(prompts, max_toks, max_batch_size=4)
print(f"  Continuous: {c_out['total_time_s']:.2f}s | {c_out['throughput_tps']:.1f} tok/s | {c_out['total_tokens']} tokens")

# ── 4. Speculative decoding ──────────────────────────────────────────────────
divider("4. Speculative Decoding (gpt2 draft → gpt2-medium target, k=4)")
from engine.speculative import SpeculativeEngine
spec_eng = SpeculativeEngine()
text, stats = spec_eng.generate(PROMPT, max_new_tokens=MAX_TOKENS, k=4)
show("speculative", text, stats)

# ── 5. Quantization ──────────────────────────────────────────────────────────
divider("5. Quantization")
from engine.quantization import QuantizationEngine

for mode in ("fp32", "int8", "w4"):
    q_eng = QuantizationEngine(mode=mode)
    text, stats = q_eng.generate(PROMPT, max_new_tokens=MAX_TOKENS)
    show(f"quant/{mode}", text, stats)

print("\n✓ All engines OK\n")
