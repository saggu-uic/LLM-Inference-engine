"""Measurement utilities: timing, memory, and result formatting."""
import time
import os
import torch
import psutil
from dataclasses import dataclass, field


@dataclass
class BenchmarkResult:
    name: str
    ttft_ms: float
    throughput_tps: float
    latency_ms: float
    memory_mb: float
    extra: dict = field(default_factory=dict)

    def __str__(self):
        extras = "  ".join(f"{k}={v}" for k, v in self.extra.items())
        return (
            f"{self.name:<28} | "
            f"TTFT {self.ttft_ms:>7.1f} ms | "
            f"{self.throughput_tps:>7.1f} tok/s | "
            f"Latency {self.latency_ms:>7.1f} ms | "
            f"Mem {self.memory_mb:>6.0f} MB"
            + (f"  [{extras}]" if extras else "")
        )


class Timer:
    def __init__(self):
        self._start = None
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start

    def split(self) -> float:
        return time.perf_counter() - self._start


class MemoryTracker:
    def __init__(self):
        self.peak_mb = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        else:
            self._rss_before = psutil.Process(os.getpid()).memory_info().rss
        return self

    def __exit__(self, *_):
        if torch.cuda.is_available():
            self.peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        else:
            rss_now = psutil.Process(os.getpid()).memory_info().rss
            self.peak_mb = max(0.0, (rss_now - self._rss_before) / 1024 ** 2)


def print_table(results: list[BenchmarkResult]):
    header = (
        f"{'Engine':<28} | {'TTFT':>12} | {'Throughput':>12} | {'Latency':>14} | {'Memory':>10}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in results:
        print(r)
    print("=" * len(header) + "\n")
