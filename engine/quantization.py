"""
Quantization: trade model weight precision for speed and memory.

GPT-2 uses transformers' Conv1D (weight shape: in×out, NOT out×in like nn.Linear).
We convert those to nn.Linear first so standard tooling can process them.

INT8  — torch.ao.quantization.quantize_dynamic
        ~2× smaller; slight accuracy drop; works on CPU without extra libs.

W4A16 — custom groupwise 4-bit weight quantization
        Weights packed INT4→INT8 (2 per byte); scales FP16 per group.
        ~4× smaller than FP32; quality close to INT8 on most tasks.
        On CPU the dequant overhead hurts; GPU (bitsandbytes/GPTQ) is much faster.
"""
import time
import torch
import torch.nn as nn
import torch.ao.quantization as tq
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Locate transformers' Conv1D regardless of where it moved across versions
_Conv1D = None
for _path in (
    "transformers.pytorch_utils.Conv1D",
    "transformers.modeling_utils.Conv1D",
    "transformers.activations.Conv1D",
):
    try:
        mod, cls = _path.rsplit(".", 1)
        import importlib
        _Conv1D = getattr(importlib.import_module(mod), cls)
        break
    except (ImportError, AttributeError):
        pass


# ---------------------------------------------------------------------------
# Helpers: Conv1D → nn.Linear conversion
# ---------------------------------------------------------------------------

def _conv1d_to_linear(module: nn.Module) -> nn.Module:
    """Replace every Conv1D with an equivalent nn.Linear (in-place, recursive)."""
    if _Conv1D is None:
        return module
    for name, child in list(module.named_children()):
        if isinstance(child, _Conv1D):
            in_f, out_f = child.weight.shape  # Conv1D: (in, out)
            lin = nn.Linear(in_f, out_f, bias=True)
            lin.weight.data = child.weight.t().contiguous()  # Linear: (out, in)
            lin.bias.data = child.bias.data
            setattr(module, name, lin)
        else:
            _conv1d_to_linear(child)
    return module


# ---------------------------------------------------------------------------
# W4A16 custom quantization
# ---------------------------------------------------------------------------

class _Linear4bit(nn.Module):
    """Drop-in nn.Linear replacement with 4-bit grouped weight quantization."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, group_size: int = 128):
        super().__init__()
        out_features, in_features = weight.shape

        pad = (group_size - in_features % group_size) % group_size
        if pad:
            weight = torch.cat([weight, weight.new_zeros(out_features, pad)], dim=1)
        in_padded = weight.shape[1]
        n_groups = in_padded // group_size

        w = weight.reshape(out_features, n_groups, group_size)
        scale = w.abs().amax(dim=-1, keepdim=True) / 7.0
        scale = scale.clamp(min=1e-8)
        q = (w / scale).round().clamp(-8, 7).to(torch.int8)

        # Pack 2 × INT4 into each INT8 byte
        packed = ((q[:, :, 0::2] & 0xF) << 4) | (q[:, :, 1::2] & 0xF)

        self.register_buffer("packed_weight", packed)          # (O, G, gs/2)
        self.register_buffer("scale", scale.to(torch.float16))
        # Register None so the buffer exists for state_dict compatibility
        self.register_buffer("bias_buf", bias.to(torch.float16) if bias is not None else None)

        self.out_features = out_features
        self.in_features = in_features
        self.in_padded = in_padded

    def _dequantize(self) -> torch.Tensor:
        packed = self.packed_weight
        q_hi = ((packed >> 4) & 0xF).to(torch.int8)
        q_lo = (packed & 0xF).to(torch.int8)
        q_hi[q_hi > 7] -= 16
        q_lo[q_lo > 7] -= 16
        O, G, half_gs = packed.shape
        q = torch.stack([q_hi, q_lo], dim=-1).reshape(O, G, half_gs * 2)
        w = (q.to(torch.float16) * self.scale).reshape(O, -1)[:, : self.in_features]
        return w  # (out, in) FP16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._dequantize()
        return nn.functional.linear(x.half(), w, self.bias_buf).to(x.dtype)


def _quantize_4bit(model: nn.Module, group_size: int = 128) -> nn.Module:
    """Replace all nn.Linear with _Linear4bit (recursive, in-place)."""
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear):
            q = _Linear4bit(
                child.weight.data,
                child.bias.data if child.bias is not None else None,
                group_size,
            )
            setattr(model, name, q)
        else:
            _quantize_4bit(child, group_size)
    return model


# ---------------------------------------------------------------------------
# Quantization engine
# ---------------------------------------------------------------------------

class QuantizationEngine:
    """Load GPT-2 in FP32, INT8, or W4A16 and compare inference."""

    MODES = ("fp32", "int8", "w4")

    def __init__(self, model_name: str = "gpt2", mode: str = "fp32", device: str | None = None):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.mode = mode
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        print(f"  Loading {model_name} in {mode.upper()} mode …")
        model = GPT2LMHeadModel.from_pretrained(model_name, dtype=torch.float32)

        if mode == "int8":
            # quantize_dynamic requires nn.Linear; convert Conv1D first
            _conv1d_to_linear(model)
            model = tq.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
            model.eval()
            # Dynamic quantisation works best on CPU
            self.device = torch.device("cpu")
        elif mode == "w4":
            _conv1d_to_linear(model)
            _quantize_4bit(model)
            model.to(self.device).eval()
        else:
            model.to(self.device).eval()

        self.model = model

    def model_size_mb(self) -> float:
        total = 0
        for p in self.model.parameters():
            total += p.nelement() * p.element_size()
        for b in self.model.buffers():
            total += b.nelement() * b.element_size()
        return total / 1024 ** 2

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        greedy: bool = True,
    ) -> tuple[str, dict]:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        t0 = time.perf_counter()

        out = self.model(input_ids, use_cache=True)
        past_kv = out.past_key_values
        last_logits = out.logits[:, -1, :]
        ttft = time.perf_counter() - t0

        generated_ids = []
        for _ in range(max_new_tokens):
            next_tok = (
                last_logits.argmax(dim=-1, keepdim=True) if greedy
                else torch.multinomial(torch.softmax(last_logits, dim=-1), 1)
            )
            generated_ids.append(next_tok.item())
            if next_tok.item() == self.tokenizer.eos_token_id:
                break
            out = self.model(next_tok, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            last_logits = out.logits[:, -1, :]

        total = time.perf_counter() - t0
        full = torch.cat([input_ids, torch.tensor([generated_ids], device=self.device)], dim=-1)
        text = self.tokenizer.decode(full[0], skip_special_tokens=True)

        return text, {
            "ttft_ms": ttft * 1000,
            "latency_ms": total * 1000,
            "new_tokens": len(generated_ids),
            "throughput_tps": len(generated_ids) / total,
            "model_size_mb": f"{self.model_size_mb():.0f} MB",
        }
