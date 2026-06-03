#!/usr/bin/env python3
"""Main 23 real-model KV validation runner.

Runs a HuggingFace causal LM (auto-downloaded) in baseline and compressed modes.
Compresses KV past tensors via RFSN TurboQuant, then decodes and compares logits.

Supports:
- --model: HuggingFace model ID (default: Qwen/Qwen2.5-0.5B-Instruct)
- --configs: comma-separated config names (e.g., k8_v3_gs64,k4_v4_gs64)
- --contexts: comma-separated token counts for long-context validation
- --tokens: number of tokens to test (default 512)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from rfsn_v10.kv_manager import RFSNTurboQuantKVManager


# Alpha pass thresholds
COSINE_MEAN_THRESHOLD = 0.995
COSINE_MIN_THRESHOLD = 0.990
TOP1_MATCH_THRESHOLD = 0.95
TOP5_OVERLAP_THRESHOLD = 0.95
PPL_DELTA_REL_THRESHOLD = 0.10
KL_DIV_THRESHOLD = 0.02


def _parse_config(name: str) -> dict[str, Any]:
    """Parse config name like 'k8_v3_gs64' -> bits and group_size."""
    if name == "baseline_fp16":
        return {"name": name, "k_bits": 16, "v_bits": 16, "group_size": 64}
    # Expected format: k{bits}_v{bits}_gs{group_size}
    parts = name.split("_")
    if len(parts) != 3:
        raise ValueError(
            f"Config '{name}' must have format k{{bits}}_v{{bits}}_gs{{group_size}}"
        )
    try:
        k_bits = int(parts[0][1:])
        v_bits = int(parts[1][1:])
        group_size = int(parts[2][2:])
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"Config '{name}' must have format k{{bits}}_v{{bits}}_gs{{group_size}}"
        ) from exc
    return {"name": name, "k_bits": k_bits, "v_bits": v_bits, "group_size": group_size}


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.reshape(-1).float()
    b_f = b.reshape(-1).float()
    return float(F.cosine_similarity(a_f, b_f, dim=0).item())


def _kl_div(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    p = F.softmax(p_logits.float(), dim=-1)
    q = F.softmax(q_logits.float(), dim=-1)
    eps = 1e-10
    kl = torch.sum(p * torch.log((p + eps) / (q + eps)))
    return float(kl.item())


def _topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    ai = set(torch.topk(a, k=k, dim=-1).indices[0].tolist())
    bi = set(torch.topk(b, k=k, dim=-1).indices[0].tolist())
    if not ai:
        return 0.0
    return float(len(ai & bi) / len(ai))


def _perplexity_for_target(logits: torch.Tensor, target_id: int) -> float:
    target = torch.tensor([target_id], device=logits.device, dtype=torch.long)
    loss = F.cross_entropy(logits.float(), target)
    return float(torch.exp(loss).item())


def _compress_decompress_past(
    past_key_values,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Compress and decompress past KV values using RFSN manager."""
    if config["name"] == "baseline_fp16":
        return past_key_values

    mgr = RFSNTurboQuantKVManager(
        k_bits=config["k_bits"],
        v_bits=config["v_bits"],
        group_size=config["group_size"],
        use_wht=True,
        use_incoherent_signs=True,
        prefer_metal_kernels=True,
        strict_metal=False,
        max_memory_gb=2.0,
    )

    rebuilt: list[tuple[torch.Tensor, torch.Tensor]] = []
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
            raise RuntimeError("Unexpected cache miss during validation")
        rk_mx, rv_mx = rec
        rk = torch.from_numpy(np.array(rk_mx))
        rv = torch.from_numpy(np.array(rv_mx))

        rk = rk[..., :dim]
        rv = rv[..., :dim]

        rk = rk.to(device=device, dtype=k_t.dtype)
        rv = rv.to(device=device, dtype=v_t.dtype)
        rebuilt.append((rk, rv))

    return tuple(rebuilt)


def _to_legacy_cache(past_key_values):
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache(), type(past_key_values)
    return past_key_values, None


def _from_legacy_cache(legacy_cache, cache_cls):
    if cache_cls is not None and hasattr(cache_cls, "from_legacy_cache"):
        return cache_cls.from_legacy_cache(legacy_cache)
    return legacy_cache


def _evaluate_config(
    config: dict[str, Any],
    *,
    model,
    tokenizer,
    past_legacy,
    cache_cls,
    decode_token: torch.Tensor,
    baseline_logits: torch.Tensor,
    baseline_ppl: float,
    device: torch.device,
) -> dict[str, Any]:
    if config["name"] != "baseline_fp16":
        past_legacy = _compress_decompress_past(past_legacy, config, device)

    past = _from_legacy_cache(past_legacy, cache_cls)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=decode_token, past_key_values=past, use_cache=True)
    dt = (time.perf_counter() - t0) * 1000.0
    logits = out.logits[:, -1, :]

    target_id = int(decode_token[0, 0].item())
    ppl = _perplexity_for_target(logits, target_id)

    cosine = _cosine(logits, baseline_logits)
    max_abs_diff = float(torch.max(torch.abs(logits - baseline_logits)).item())
    top1_match = float(
        torch.argmax(logits, dim=-1).item() == torch.argmax(baseline_logits, dim=-1).item()
    )
    top5 = _topk_overlap(logits, baseline_logits, k=5)
    kl = _kl_div(baseline_logits, logits)

    return {
        "name": config["name"],
        "k_bits": config["k_bits"],
        "v_bits": config["v_bits"],
        "group_size": config["group_size"],
        "logit_cosine_mean": cosine,
        "logit_cosine_min": cosine,
        "logit_max_abs_diff": max_abs_diff,
        "top1_match_rate": top1_match,
        "top5_overlap_mean": top5,
        "perplexity_delta": float(ppl - baseline_ppl),
        "kl_divergence_mean": kl,
        "latency_ms": dt,
        "route_used": "retrieve" if config["name"] != "baseline_fp16" else "baseline_fp16",
    }


def _determine_status(result: dict[str, Any], *, baseline_ppl: float = 1.0) -> str:
    """Apply alpha pass thresholds honestly."""
    if result["name"] == "baseline_fp16":
        return "reference"

    ppl_delta_rel = abs(result["perplexity_delta"]) / max(abs(baseline_ppl), 1e-8)

    if result["logit_cosine_mean"] < COSINE_MEAN_THRESHOLD:
        return "fail"
    if result["logit_cosine_min"] < COSINE_MIN_THRESHOLD:
        return "fail"
    if result["top1_match_rate"] < TOP1_MATCH_THRESHOLD:
        return "fail"
    if result["top5_overlap_mean"] < TOP5_OVERLAP_THRESHOLD:
        return "fail"
    if ppl_delta_rel > PPL_DELTA_REL_THRESHOLD:
        return "fail"
    if result["kl_divergence_mean"] > KL_DIV_THRESHOLD:
        return "fail"
    return "pass"


def _run_real_model_validation(
    model_id: str,
    tokens: int,
    configs: list[dict[str, Any]],
    device: torch.device,
    out_path: Path,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    dtype = torch.float16 if device.type == "mps" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()

    # Build a prompt of roughly the target length
    prompt_text = "The quick brown fox jumps over the lazy dog. " * 200
    inputs = tokenizer(prompt_text, return_tensors="pt")
    if inputs["input_ids"].shape[1] < tokens:
        print(
            f"WARNING: requested {tokens} tokens but prompt only has "
            f"{inputs['input_ids'].shape[1]}. Testing with available tokens."
        )
    input_ids = inputs["input_ids"][:, :tokens]
    if input_ids.shape[1] < 2:
        raise ValueError("Need at least 2 tokens")

    input_ids = input_ids.to(device)
    context_ids = input_ids[:, :-1]
    decode_token = input_ids[:, -1:]

    with torch.no_grad():
        baseline_ctx = model(input_ids=context_ids, use_cache=True)
        baseline_out = model(
            input_ids=decode_token, past_key_values=baseline_ctx.past_key_values, use_cache=True
        )
    baseline_logits = baseline_out.logits[:, -1, :]
    baseline_target = int(decode_token[0, 0].item())
    baseline_ppl = _perplexity_for_target(baseline_logits, baseline_target)
    baseline_legacy, baseline_cache_cls = _to_legacy_cache(baseline_ctx.past_key_values)

    config_results: list[dict[str, Any]] = []
    for config in configs:
        print(f"  Evaluating config: {config['name']} ...")
        result = _evaluate_config(
            config,
            model=model,
            tokenizer=tokenizer,
            past_legacy=baseline_legacy,
            cache_cls=baseline_cache_cls,
            decode_token=decode_token,
            baseline_logits=baseline_logits,
            baseline_ppl=baseline_ppl,
            device=device,
        )
        result["status"] = _determine_status(result, baseline_ppl=baseline_ppl)
        config_results.append(result)
        ppl_rel = abs(result["perplexity_delta"]) / max(abs(baseline_ppl), 1e-8)
        print(
            f"    cosine={result['logit_cosine_mean']:.6f} "
            f"top1={result['top1_match_rate']:.3f} "
            f"top5={result['top5_overlap_mean']:.3f} "
            f"ppl_delta={result['perplexity_delta']:.6f} "
            f"ppl_rel={ppl_rel:.4f} "
            f"kl={result['kl_divergence_mean']:.6f} "
            f"status={result['status']}"
        )

    payload: dict[str, Any] = {
        "release": "main23",
        "validation_class": "real_non_random_model_validation",
        "model": model_id,
        "hardware": _get_hardware_info(),
        "tokens_tested": tokens,
        "configs": config_results,
        "sparse_enabled": False,
        "notes": [
            "Real non-random model validation executed.",
            "Sparse decode is disabled by default.",
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote real model validation to {out_path}")
    return payload


def _run_long_context_validation(
    model_id: str,
    contexts: list[int],
    configs: list[dict[str, Any]],
    device: torch.device,
    out_path: Path,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    dtype = torch.float16 if device.type == "mps" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()

    prompt_text = "The quick brown fox jumps over the lazy dog. " * 500
    all_input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]

    context_entries: list[dict[str, Any]] = []
    for ctx_tokens in contexts:
        print(f"Long-context validation: {ctx_tokens} tokens ...")
        input_ids = all_input_ids[:, :ctx_tokens].to(device)
        if input_ids.shape[1] < 2:
            continue
        context_ids = input_ids[:, :-1]
        decode_token = input_ids[:, -1:]

        with torch.no_grad():
            baseline_ctx = model(input_ids=context_ids, use_cache=True)
            baseline_out = model(
                input_ids=decode_token,
                past_key_values=baseline_ctx.past_key_values,
                use_cache=True,
            )
        baseline_logits = baseline_out.logits[:, -1, :]
        baseline_target = int(decode_token[0, 0].item())
        baseline_ppl = _perplexity_for_target(baseline_logits, baseline_target)
        baseline_legacy, baseline_cache_cls = _to_legacy_cache(baseline_ctx.past_key_values)

        config_results: list[dict[str, Any]] = []
        for config in configs:
            try:
                result = _evaluate_config(
                    config,
                    model=model,
                    tokenizer=tokenizer,
                    past_legacy=baseline_legacy,
                    cache_cls=baseline_cache_cls,
                    decode_token=decode_token,
                    baseline_logits=baseline_logits,
                    baseline_ppl=baseline_ppl,
                    device=device,
                )
                result["status"] = _determine_status(result, baseline_ppl=baseline_ppl)
                result["oom"] = False
            except Exception as e:
                msg = str(e).lower()
                if "out of memory" in msg or "no memory" in msg or "mps allocator" in msg:
                    result = {"name": config["name"], "oom": True, "status": "oom"}
                else:
                    raise
            config_results.append(result)

        context_entries.append({
            "tokens": ctx_tokens,
            "configs": config_results,
        })

    # Determine best configs from results
    def _best_quality(ctxs):
        best = ""
        best_cos = -1.0
        for ctx in ctxs:
            for c in ctx["configs"]:
                if c.get("oom"):
                    continue
                cos = c.get("logit_cosine_mean", -1.0)
                if cos > best_cos:
                    best_cos = cos
                    best = c["name"]
        return best

    def _best_memory(ctxs):
        # Prefer highest compression (lowest bits) among passing configs
        best = ""
        best_score = float("inf")
        for ctx in ctxs:
            for c in ctx["configs"]:
                if c.get("oom") or c.get("status") != "pass":
                    continue
                score = c.get("k_bits", 16) + c.get("v_bits", 16)
                if score < best_score:
                    best_score = score
                    best = c["name"]
        return best

    def _recommended(ctxs):
        # Default to k8_v3 if it passes, else k4_v4, else baseline
        for ctx in ctxs:
            for c in ctx["configs"]:
                if c.get("name") == "k8_v3_gs64" and c.get("status") == "pass":
                    return "k8_v3_gs64"
        for ctx in ctxs:
            for c in ctx["configs"]:
                if c.get("name") == "k4_v4_gs64" and c.get("status") == "pass":
                    return "k4_v4_gs64"
        return "baseline_fp16"

    payload: dict[str, Any] = {
        "release": "main23",
        "model": model_id,
        "contexts": context_entries,
        "summary": {
            "best_quality_config": _best_quality(context_entries),
            "best_memory_config": _best_memory(context_entries),
            "recommended_default": _recommended(context_entries),
            "production_ready": False,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote long-context validation to {out_path}")
    return payload


def _get_hardware_info() -> dict[str, Any]:
    import platform
    import subprocess

    mlx_version = "unknown"
    try:
        import mlx
        mlx_version = mlx.__version__
    except Exception:
        pass

    chip = "unknown"
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=False,
        )
        chip = result.stdout.strip() or "unknown"
    except Exception:
        pass

    ram_gb = 16
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / (1024 ** 3))
    except Exception:
        pass

    return {
        "chip": chip,
        "ram_gb": ram_gb,
        "os": platform.system(),
        "mlx_version": mlx_version,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate KV quality against a real HuggingFace model"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model ID to validate against",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=512,
        help="Number of tokens to test",
    )
    parser.add_argument(
        "--configs",
        default="baseline_fp16,k8_v3_gs64,k4_v4_gs64",
        help="Comma-separated config names to test",
    )
    parser.add_argument(
        "--contexts",
        default="",
        help="Comma-separated token counts for long-context validation (e.g., 512,1024,2048)",
    )
    parser.add_argument(
        "--out",
        default="artifacts/proof/main23/real_model_validation.json",
        help="Output JSON path for real-model validation",
    )
    parser.add_argument(
        "--long-context-out",
        default="artifacts/proof/main23/long_context_validation.json",
        help="Output JSON path for long-context validation",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading the model from HuggingFace",
    )
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    configs = [_parse_config(name) for name in config_names]

    # Real-model validation
    print(f"Running real-model validation: model={args.model}, tokens={args.tokens}")
    payload = _run_real_model_validation(
        model_id=args.model,
        tokens=args.tokens,
        configs=configs,
        device=device,
        out_path=Path(args.out),
        trust_remote_code=args.trust_remote_code,
    )
    exit_code = 0
    if any(c.get("status") == "fail" for c in payload.get("configs", [])):
        exit_code = 1

    # Long-context validation
    if args.contexts:
        context_tokens = [int(c.strip()) for c in args.contexts.split(",") if c.strip()]
        _run_long_context_validation(
            model_id=args.model,
            contexts=context_tokens,
            configs=configs,
            device=device,
            out_path=Path(args.long_context_out),
            trust_remote_code=args.trust_remote_code,
        )

    if exit_code != 0:
        print("FAIL: one or more configs did not meet quality thresholds.", file=sys.stderr)
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
