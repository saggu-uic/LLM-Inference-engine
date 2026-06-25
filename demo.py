"""
Quick smoke-test: one generation from each engine. No benchmarking.

Usage:  python demo.py
        python demo.py --skip-spec   # skip if gpt2-medium not cached
"""
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--skip-spec", action="store_true")
args = parser.parse_args()

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
            print(f"  {k:<22}: {v:.2f}")
        else:
            print(f"  {k:<22}: {v}")


# ── 1. Baseline ──────────────────────────────────────────────────────────────
divider("1. Baseline (no KV cache)")
from engine.baseline import BaselineEngine
eng = BaselineEngine()
text, stats = eng.generate(PROMPT, max_new_tokens=MAX_TOKENS)
show("baseline", text, stats)

# ── 2. KV Cache ──────────────────────────────────────────────────────────────
divider("2. KV Cache + Prefix Reuse")
from engine.kv_cache import KVCacheEngine
eng = KVCacheEngine()
text, stats = eng.generate(PROMPT, max_new_tokens=MAX_TOKENS)
show("kv_cache", text, stats)

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
print(f"\n  Static    : {s_out['total_time_s']:.2f}s | {s_out['throughput_tps']:.1f} tok/s | {s_out['total_tokens']} tokens")

cont_eng = ContinuousBatchingEngine()
c_out = cont_eng.run(prompts, max_toks, max_batch_size=4)
print(f"  Continuous: {c_out['total_time_s']:.2f}s | {c_out['throughput_tps']:.1f} tok/s | {c_out['total_tokens']} tokens")

# ── 4. Paged attention ───────────────────────────────────────────────────────
divider("4. Paged Attention")
from engine.paged_attention import PagedAttentionEngine

paged_eng = PagedAttentionEngine()
text, stats = paged_eng.generate(PROMPT, max_new_tokens=MAX_TOKENS)
show("paged_attention", text, stats)

# Memory comparison: 8 requests with heterogeneous lengths
DEMO_PROMPTS = [
    "Tell me a short story about a cat.",
    "What is the capital of France?",
    "Write a haiku about autumn leaves.",
    "Explain quantum entanglement.",
    "Describe the water cycle.",
    "What are the primary colors?",
    "Write a limerick about a programmer.",
    "How does photosynthesis work?",
]
DEMO_MAX_TOKENS = [60, 15, 20, 40, 20, 10, 50, 35]

mc = paged_eng.memory_comparison(DEMO_PROMPTS, DEMO_MAX_TOKENS)
print(f"\n  Memory comparison ({mc['num_sequences']} sequences, max_new={mc['max_new_tokens']} tok):")
print(f"    Naive pre-allocation : {mc['naive_mb']:.2f} MB")
print(f"    Paged allocation     : {mc['paged_mb']:.2f} MB")
print(f"    Savings              : {mc['savings_mb']:.2f} MB  ({mc['savings_pct']:.1f}%)")

# ── 5. Speculative decoding ──────────────────────────────────────────────────
if not args.skip_spec:
    divider("5. Speculative Decoding (gpt2 draft → gpt2-medium target, k=4)")
    from engine.speculative import SpeculativeEngine
    spec_eng = SpeculativeEngine()
    text, stats = spec_eng.generate(PROMPT, max_new_tokens=MAX_TOKENS, k=4)
    show("speculative", text, stats)

print("\n✓ All engines OK\n")
