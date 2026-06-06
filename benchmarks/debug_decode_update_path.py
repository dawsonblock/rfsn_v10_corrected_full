#!/usr/bin/env python3
"""Decode-state trace diagnostic for RFSN v10.

Traces the decode/update path step-by-step to identify whether quantized
decode corruption starts immediately or accumulates over steps.

Tests:
  k8_v5_gs64, k8_v5_gs32, turbo_polar, adaptive, experimental_hybrid

Prompt lengths: 128, 512
Decode steps: 0, 1, 2, 4, 8, 16, 32

Output:
  artifacts/proof/experimental/decode_update_trace.json

Each trace row captures KV cache shape, position metadata, and per-step
logit quality metrics to isolate decode corruption patterns.
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as functional
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from rfsn_v10.kv_manager import RFSNTurboQuantKVManager
from rfsn_v10.quantization.kv_quant_manager import QuantizedKVManager
from rfsn_v10.quantization.turbo_polar_kv_manager import TurboPolarKVManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.reshape(-1).float()
    b_f = b.reshape(-1).float()
    if torch.isnan(a_f).any() or torch.isinf(a_f).any():
        return float("nan")
    if torch.isnan(b_f).any() or torch.isinf(b_f).any():
        return float("nan")
    return float(functional.cosine_similarity(a_f, b_f, dim=0).item())


def _kl_div(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    p = functional.softmax(p_logits.float(), dim=-1)
    q = functional.softmax(q_logits.float(), dim=-1)
    eps = 1e-10
    kl = torch.sum(p * torch.log((p + eps) / (q + eps)))
    return float(kl.item())


def _topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int = 5) -> float:
    ai = set(torch.topk(a, k=k, dim=-1).indices[0].tolist())
    bi = set(torch.topk(b, k=k, dim=-1).indices[0].tolist())
    if not ai:
        return 0.0
    return float(len(ai & bi) / len(ai))


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------


def _compress_cache_stable(
    past_key_values, k_bits: int, v_bits: int, group_size: int, device: torch.device,
):
    """Compress via stable RFSN quantization and return DynamicCache."""
    compressed = []
    with tempfile.TemporaryDirectory(prefix="rfsn_decode_") as tmpdir:
        mgr = RFSNTurboQuantKVManager(
            k_bits=k_bits, v_bits=v_bits, group_size=group_size,
            use_wht=True, use_incoherent_signs=True,
            prefer_metal_kernels=True, strict_metal=False,
            max_memory_gb=2.0, cache_dir=tmpdir,
        )
        for layer_idx, (k_t, v_t) in enumerate(past_key_values):
            k_np = k_t.detach().to("cpu", dtype=torch.float32).numpy()
            v_np = v_t.detach().to("cpu", dtype=torch.float32).numpy()
            bsz, heads, seq, dim = k_np.shape
            dim_padded = int(math.ceil(dim / 64.0) * 64)
            if dim_padded != dim:
                pad = dim_padded - dim
                k_np = np.pad(k_np, ((0, 0), (0, 0), (0, 0), (0, pad)))
                v_np = np.pad(v_np, ((0, 0), (0, 0), (0, 0), (0, pad)))
            k_mx = mx.array(k_np)
            v_mx = mx.array(v_np)
            key = f"layer_{layer_idx}"
            mgr.store(key, k_mx, v_mx, token_count=seq)
            rec = mgr.retrieve(key, out_dtype=mx.float32)
            if rec is None:
                raise RuntimeError("Cache miss")
            rk_mx, rv_mx = rec
            rk = torch.from_numpy(np.array(rk_mx))[:, :, :, :dim]
            rv = torch.from_numpy(np.array(rv_mx))[:, :, :, :dim]
            rk = rk.to(device=device, dtype=k_t.dtype)
            rv = rv.to(device=device, dtype=v_t.dtype)
            compressed.append((rk, rv))
    return compressed


def _compress_cache_experimental(
    past_key_values, cfg: dict[str, Any], device: torch.device,
):
    """Compress via experimental quantization managers."""
    mode = cfg.get("mode", "hybrid_polar_cartesian")
    if mode == "turbo_polar":
        mgr = TurboPolarKVManager(
            feature_dim=cfg.get("feature_dim", 64),
            k_angle_bits=cfg.get("k_angle_bits", 5),
            k_radius_bits=cfg.get("k_radius_bits", 8),
            v_bits=cfg.get("v_bits", 6),
            group_size=cfg.get("group_size", 64),
            adaptive_angle_range=cfg.get("adaptive_angle_range", False),
        )
    else:
        mgr = QuantizedKVManager(
            mode="hybrid_polar_cartesian",
            feature_dim=cfg.get("feature_dim", 64),
            polar_ratio=cfg.get("polar_ratio", 0.65),
            polar_levels=cfg.get("polar_levels", 4),
            k_angle_bits=cfg.get("k_angle_bits", 5),
            k_radius_bits=cfg.get("k_radius_bits", 8),
            v_angle_bits=cfg.get("v_angle_bits", 4),
            v_radius_bits=cfg.get("v_radius_bits", 6),
            cartesian_bits=cfg.get("cartesian_bits", 6),
            group_size=cfg.get("group_size", 64),
        )
    compressed = []
    for layer_idx, (k_t, v_t) in enumerate(past_key_values):
        k_np = k_t.detach().to("cpu", dtype=torch.float32).numpy()
        v_np = v_t.detach().to("cpu", dtype=torch.float32).numpy()
        bsz, heads, seq, dim = k_np.shape
        dim_padded = int(math.ceil(dim / 64.0) * 64)
        if dim_padded != dim:
            pad = dim_padded - dim
            k_np = np.pad(k_np, ((0, 0), (0, 0), (0, 0), (0, pad)))
            v_np = np.pad(v_np, ((0, 0), (0, 0), (0, 0), (0, pad)))
        k_mx = mx.array(k_np)
        v_mx = mx.array(v_np)
        packet = mgr.quantize(k_mx, v_mx)
        rk_mx, rv_mx = mgr.dequantize(packet)
        rk = torch.from_numpy(np.array(rk_mx))[:, :, :, :dim]
        rv = torch.from_numpy(np.array(rv_mx))[:, :, :, :dim]
        rk = rk.to(device=device, dtype=k_t.dtype)
        rv = rv.to(device=device, dtype=v_t.dtype)
        compressed.append((rk, rv))
    return compressed


# ---------------------------------------------------------------------------
# Config registry
# ---------------------------------------------------------------------------

CONFIGS: dict[str, dict[str, Any]] = {
    "k8_v5_gs64": {"family": "stable", "k_bits": 8, "v_bits": 5, "group_size": 64},
    "k8_v5_gs32": {"family": "stable", "k_bits": 8, "v_bits": 5, "group_size": 32},
    "turbo_polar": {
        "family": "experimental", "mode": "turbo_polar",
        "feature_dim": 64, "k_angle_bits": 5, "k_radius_bits": 8,
        "v_bits": 6, "group_size": 64,
    },
    "adaptive": {
        "family": "experimental", "mode": "turbo_polar",
        "feature_dim": 64, "k_angle_bits": 5, "k_radius_bits": 8,
        "v_bits": 6, "group_size": 64, "adaptive_angle_range": True,
    },
    "experimental_hybrid": {
        "family": "experimental", "mode": "hybrid_polar_cartesian",
        "feature_dim": 64, "polar_ratio": 0.65, "polar_levels": 4,
        "k_angle_bits": 5, "k_radius_bits": 8,
        "v_angle_bits": 4, "v_radius_bits": 6,
        "cartesian_bits": 6, "group_size": 64,
    },
}


# ---------------------------------------------------------------------------
# Trace runner
# ---------------------------------------------------------------------------


def trace_decode_steps(
    model_name: str,
    cfg_name: str,
    prompt_tokens: int,
    decode_steps: list[int],
    device: torch.device,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Run decode steps and trace KV cache state at each checkpoint."""
    cfg = CONFIGS[cfg_name]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map="cpu", trust_remote_code=True,
    )
    model.eval()
    torch.manual_seed(seed)

    # Build prompt
    dummy_text = "The quick brown fox jumps over the lazy dog. " * 200
    dummy_ids = tokenizer.encode(dummy_text, add_special_tokens=False)
    if prompt_tokens > len(dummy_ids):
        repeated = (dummy_ids * ((prompt_tokens // len(dummy_ids)) + 1))[:prompt_tokens]
        prompt_str = tokenizer.decode(repeated)
    else:
        prompt_str = tokenizer.decode(dummy_ids[:prompt_tokens])

    prompt_ids = tokenizer.encode(prompt_str, return_tensors="pt", truncation=True)
    prompt_ids = prompt_ids.to(device)
    actual_prompt_len = prompt_ids.shape[1]

    # FP16 prefill
    with torch.no_grad():
        out_fp16 = model(input_ids=prompt_ids, past_key_values=None, use_cache=True)
    fp16_past = out_fp16.past_key_values
    first_tok = int(torch.argmax(out_fp16.logits[:, -1, :], dim=-1).item())

    # Compress prefill cache
    if hasattr(fp16_past, "to_legacy_cache"):
        past_list = list(fp16_past.to_legacy_cache())
    else:
        past_list = list(fp16_past)

    if cfg["family"] == "stable":
        compressed_list = _compress_cache_stable(
            past_list, cfg["k_bits"], cfg["v_bits"], cfg["group_size"], device,
        )
    else:
        compressed_list = _compress_cache_experimental(past_list, cfg, device)

    compressed_past = DynamicCache.from_legacy_cache(tuple(compressed_list))

    # Run decode steps on both FP16 and compressed paths
    max_steps = max(decode_steps)
    fp16_decode_past = fp16_past
    quant_decode_past = compressed_past

    results: list[dict[str, Any]] = []
    next_tok_fp16 = first_tok
    next_tok_quant = first_tok

    for step in range(max_steps + 1):
        # Check if this step is a checkpoint
        if step in decode_steps:
            # Get current cache shapes
            if hasattr(fp16_decode_past, "to_legacy_cache"):
                fp16_legacy = list(fp16_decode_past.to_legacy_cache())
            else:
                fp16_legacy = list(fp16_decode_past)
            if hasattr(quant_decode_past, "to_legacy_cache"):
                quant_legacy = list(quant_decode_past.to_legacy_cache())
            else:
                quant_legacy = list(quant_decode_past)

            k_shape = list(fp16_legacy[0][0].shape) if fp16_legacy else []
            v_shape = list(fp16_legacy[0][1].shape) if fp16_legacy else []
            kv_len = k_shape[2] if len(k_shape) > 2 else 0

            # Compare logits at this step
            next_ids_fp16 = torch.tensor([[next_tok_fp16]], device=device)
            next_ids_quant = torch.tensor([[next_tok_quant]], device=device)

            with torch.no_grad():
                out_fp16_step = model(
                    input_ids=next_ids_fp16,
                    past_key_values=fp16_decode_past, use_cache=True,
                )
                out_quant_step = model(
                    input_ids=next_ids_quant,
                    past_key_values=quant_decode_past, use_cache=True,
                )

            logit_fp16 = out_fp16_step.logits[:, -1, :]
            logit_quant = out_quant_step.logits[:, -1, :]

            cosine = _cosine(logit_fp16, logit_quant)
            top5 = _topk_overlap(logit_fp16, logit_quant, k=5)
            kl = _kl_div(logit_fp16, logit_quant)

            status = "pass" if cosine >= 0.99 and top5 >= 0.8 else "degraded"

            results.append({
                "config": cfg_name,
                "prompt_tokens": actual_prompt_len,
                "decode_step": step,
                "kv_len_before": kv_len,
                "kv_len_after": kv_len + 1,
                "position_id": actual_prompt_len + step,
                "cache_position": actual_prompt_len + step,
                "k_shape": k_shape,
                "v_shape": v_shape,
                "logit_cosine": cosine,
                "top5_overlap": top5,
                "kl": kl,
                "status": status,
            })

            # Update past for continuation
            fp16_decode_past = out_fp16_step.past_key_values
            quant_decode_past = out_quant_step.past_key_values
            next_tok_fp16 = int(torch.argmax(logit_fp16, dim=-1).item())
            next_tok_quant = int(torch.argmax(logit_quant, dim=-1).item())
        else:
            # Non-checkpoint step: just advance both
            next_ids_fp16 = torch.tensor([[next_tok_fp16]], device=device)
            next_ids_quant = torch.tensor([[next_tok_quant]], device=device)
            with torch.no_grad():
                out_fp16_step = model(
                    input_ids=next_ids_fp16,
                    past_key_values=fp16_decode_past, use_cache=True,
                )
                out_quant_step = model(
                    input_ids=next_ids_quant,
                    past_key_values=quant_decode_past, use_cache=True,
                )
            fp16_decode_past = out_fp16_step.past_key_values
            quant_decode_past = out_quant_step.past_key_values
            logit_fp16 = out_fp16_step.logits[:, -1, :]
            logit_quant = out_quant_step.logits[:, -1, :]
            next_tok_fp16 = int(torch.argmax(logit_fp16, dim=-1).item())
            next_tok_quant = int(torch.argmax(logit_quant, dim=-1).item())

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode-state trace diagnostic"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--configs", nargs="+",
        default=["k8_v5_gs64", "k8_v5_gs32", "turbo_polar", "adaptive", "experimental_hybrid"],
    )
    parser.add_argument("--prompt-lengths", nargs="+", type=int, default=[128, 512])
    parser.add_argument(
        "--decode-steps", nargs="+", type=int, default=[0, 1, 2, 4, 8, 16, 32],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out", default="artifacts/proof/experimental/decode_update_trace.json",
    )
    args = parser.parse_args()

    device = torch.device("cpu")

    all_traces: list[dict[str, Any]] = []
    for cfg_name in args.configs:
        for length in args.prompt_lengths:
            print(f"Tracing {cfg_name} @ {length} tokens ...")
            try:
                traces = trace_decode_steps(
                    args.model, cfg_name, length,
                    args.decode_steps, device, args.seed,
                )
                all_traces.extend(traces)
                for t in traces:
                    print(
                        f"  step={t['decode_step']:2d} "
                        f"cosine={t['logit_cosine']:.4f} "
                        f"top5={t['top5_overlap']:.2f} "
                        f"status={t['status']}"
                    )
            except Exception as exc:
                print(f"  FAILED: {exc}")
                all_traces.append({
                    "config": cfg_name,
                    "prompt_tokens": length,
                    "error": str(exc),
                })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "release": "experimental",
                "model": args.model,
                "seed": args.seed,
                "decode_steps_checked": args.decode_steps,
                "traces": all_traces,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote trace to {out_path}")


if __name__ == "__main__":
    main()
