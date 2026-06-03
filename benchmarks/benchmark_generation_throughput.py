#!/usr/bin/env python3
"""Generation throughput benchmark for RFSN v10 Main 26.

Measures:
  - Tokens/second (decode throughput)
  - Time-to-first-token latency (TTFT)
  - Per-token decode latency (p50, p90, p99)
  - Peak memory usage (MPS / CPU RSS)

For each config, runs `--repeats` warmup + timed trials and reports
aggregated statistics in a JSON summary.

Usage:
  python benchmarks/benchmark_generation_throughput.py \\
      --model Qwen/Qwen2.5-0.5B-Instruct \\
      --tokens 256 --decode 64 --repeats 5 \\
      --out artifacts/proof/main26/generation_throughput.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Hardware info
# ---------------------------------------------------------------------------

def _get_hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        import platform
        info["platform"] = platform.platform()
        info["python"] = platform.python_version()
        info["torch"] = torch.__version__
    except Exception:
        pass
    if torch.backends.mps.is_available():
        info["device"] = "mps"
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
            )
            if result.returncode == 0:
                info["memory_bytes"] = int(result.stdout.strip())
        except Exception:
            pass
    else:
        info["device"] = "cpu"
    return info


# ---------------------------------------------------------------------------
# KV cache helpers
# ---------------------------------------------------------------------------

def _to_legacy_cache(pkv: Any) -> tuple[list, type]:
    if pkv is None:
        return [], type(None)
    if hasattr(pkv, "to_legacy_cache"):
        return pkv.to_legacy_cache(), type(pkv)
    return list(pkv), type(pkv)


def _from_legacy_cache(legacy: list, cache_cls: type) -> Any:
    if not legacy:
        return None
    if cache_cls is type(None) or cache_cls is list:
        return legacy
    try:
        return cache_cls.from_legacy_cache(legacy)
    except Exception:
        return legacy


def _clone_legacy_cache(legacy: list) -> list:
    cloned = []
    for layer in legacy:
        if isinstance(layer, (tuple, list)):
            cloned.append(tuple(t.clone() for t in layer))
        else:
            cloned.append(layer)
    return cloned


def _compress_past(past_legacy: list, config: dict, device: torch.device) -> list:
    from rfsn_v10.kv_manager import RFSNTurboQuantKVManager
    import mlx.core as mx
    import numpy as np

    k_bits = config["k_bits"]
    v_bits = config["v_bits"]
    group_size = config["group_size"]

    compressed = []
    for k, v in past_legacy:
        k_np = k.float().cpu().numpy()
        v_np = v.float().cpu().numpy()
        mgr = RFSNTurboQuantKVManager(k_bits=k_bits, v_bits=v_bits, group_size=group_size)
        token_count = k.shape[2]  # [B, H, T, D]
        mgr.store(
            skill_pattern="throughput",
            keys=mx.array(k_np),
            values=mx.array(v_np),
            token_count=token_count,
        )
        rk, rv = mgr.retrieve(skill_pattern="throughput")
        k_out = torch.from_numpy(np.array(rk)).to(device=device, dtype=k.dtype)
        v_out = torch.from_numpy(np.array(rv)).to(device=device, dtype=v.dtype)
        compressed.append((k_out, v_out))
    return compressed


# ---------------------------------------------------------------------------
# Memory snapshot
# ---------------------------------------------------------------------------

def _peak_memory_mb() -> float:
    """Returns peak resident set size in MB, or NaN if unavailable."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # On macOS ru_maxrss is in bytes; on Linux it's in kilobytes
        if sys.platform == "darwin":
            return usage.ru_maxrss / (1024 * 1024)
        return usage.ru_maxrss / 1024
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Config registry
# ---------------------------------------------------------------------------

_CONFIG_REGISTRY: dict[str, dict[str, Any]] = {
    "baseline_fp16": {"name": "baseline_fp16", "k_bits": 16, "v_bits": 16, "group_size": 64},
    "k8_v3_gs64": {"name": "k8_v3_gs64", "k_bits": 8, "v_bits": 3, "group_size": 64},
    "k8_v4_gs64": {"name": "k8_v4_gs64", "k_bits": 8, "v_bits": 4, "group_size": 64},
    "k8_v5_gs64": {"name": "k8_v5_gs64", "k_bits": 8, "v_bits": 5, "group_size": 64},
    "k8_v4_gs32": {"name": "k8_v4_gs32", "k_bits": 8, "v_bits": 4, "group_size": 32},
    "k8_v5_gs32": {"name": "k8_v5_gs32", "k_bits": 8, "v_bits": 5, "group_size": 32},
    "k6_v6_gs64": {"name": "k6_v6_gs64", "k_bits": 6, "v_bits": 6, "group_size": 64},
    "k4_v4_gs64": {"name": "k4_v4_gs64", "k_bits": 4, "v_bits": 4, "group_size": 64},
}


def _parse_config(name: str) -> dict[str, Any]:
    if name in _CONFIG_REGISTRY:
        return _CONFIG_REGISTRY[name]
    parts = name.split("_")
    cfg: dict[str, Any] = {"name": name, "k_bits": 8, "v_bits": 4, "group_size": 64}
    for p in parts:
        if p.startswith("k") and p[1:].isdigit():
            cfg["k_bits"] = int(p[1:])
        elif p.startswith("v") and p[1:].isdigit():
            cfg["v_bits"] = int(p[1:])
        elif p.startswith("gs") and p[2:].isdigit():
            cfg["group_size"] = int(p[2:])
    return cfg


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------

def _percentile(data: list[float], pct: float) -> float:
    if not data:
        return float("nan")
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * pct / 100.0
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return sorted_data[lo]
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (k - lo)


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def _run_trial(
    model,
    input_ids: torch.Tensor,
    past_legacy: list | None,
    cache_cls: type | None,
    n_decode: int,
) -> dict[str, float]:
    """Run one timed greedy decode trial. Returns timing stats."""
    if past_legacy is not None:
        past = _from_legacy_cache(past_legacy, cache_cls)
        ctx_input = input_ids[:, -1:]
    else:
        past = None
        ctx_input = input_ids

    # TTFT: time to produce first token
    t_start = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=ctx_input, past_key_values=past, use_cache=True)
    past = out.past_key_values
    t_first = time.perf_counter()
    ttft_ms = (t_first - t_start) * 1000.0

    # Decode remaining tokens
    per_token_ms: list[float] = []
    logits = out.logits[:, -1, :]
    for _ in range(n_decode):
        next_tok = torch.argmax(logits, dim=-1, keepdim=True)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(
                input_ids=next_tok, past_key_values=past, use_cache=True
            )
        t1 = time.perf_counter()
        per_token_ms.append((t1 - t0) * 1000.0)
        past = out.past_key_values
        logits = out.logits[:, -1, :]

    total_decode_ms = sum(per_token_ms)
    tokens_per_sec = (n_decode / total_decode_ms * 1000.0) if total_decode_ms > 0 else float("nan")

    return {
        "ttft_ms": ttft_ms,
        "total_decode_ms": total_decode_ms,
        "tokens_per_sec": tokens_per_sec,
        "p50_token_ms": _percentile(per_token_ms, 50),
        "p90_token_ms": _percentile(per_token_ms, 90),
        "p99_token_ms": _percentile(per_token_ms, 99),
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

_BENCH_PROMPT = (
    "The transformer architecture revolutionised natural language processing "
    "by enabling parallel computation over sequence positions. Self-attention "
    "allows each token to attend to every other token in the sequence. "
    "Positional encodings provide order information. Layer normalisation "
    "stabilises training. Feed-forward networks expand and contract the "
    "hidden dimension. These components combine to form a powerful model. "
)


def _run_benchmark(
    model_id: str,
    tokens: int,
    n_decode: int,
    repeats: int,
    warmup: int,
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
        model_id, torch_dtype=dtype, trust_remote_code=trust_remote_code
    )
    model.to(device)
    model.eval()

    prompt = _BENCH_PROMPT * max(1, tokens // len(_BENCH_PROMPT.split()) + 1)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"][:, :tokens].to(device)
    print(f"Context tokens: {input_ids.shape[1]}, decode steps: {n_decode}")

    # Pre-compute context KV cache once
    with torch.no_grad():
        ctx_out = model(input_ids=input_ids, use_cache=True)
    ctx_legacy, ctx_cache_cls = _to_legacy_cache(ctx_out.past_key_values)

    def _mean(vals: list[float]) -> float:
        finite = [v for v in vals if math.isfinite(v)]
        return sum(finite) / len(finite) if finite else float("nan")

    config_results: list[dict[str, Any]] = []

    for config in configs:
        print(f"  Benchmarking {config['name']} (warmup={warmup}, repeats={repeats}) ...")
        mem_before = _peak_memory_mb()

        # Build per-trial past caches
        def _get_past() -> tuple[list | None, type | None]:
            if config["name"] == "baseline_fp16":
                return _clone_legacy_cache(ctx_legacy), ctx_cache_cls
            return (
                _compress_past(_clone_legacy_cache(ctx_legacy), config, device),
                ctx_cache_cls,
            )

        # Warmup
        for _ in range(warmup):
            past_w, cls_w = _get_past()
            _run_trial(model, input_ids, past_w, cls_w, n_decode)

        # Timed trials
        trial_results: list[dict[str, float]] = []
        for _ in range(repeats):
            past_r, cls_r = _get_past()
            trial_results.append(
                _run_trial(model, input_ids, past_r, cls_r, n_decode)
            )

        mem_after = _peak_memory_mb()
        peak_mem_delta_mb = (
            mem_after - mem_before if math.isfinite(mem_after) else float("nan")
        )

        agg: dict[str, Any] = {
            "name": config["name"],
            "k_bits": config["k_bits"],
            "v_bits": config["v_bits"],
            "group_size": config["group_size"],
            "repeats": repeats,
            "ttft_ms_mean": _mean([r["ttft_ms"] for r in trial_results]),
            "tokens_per_sec_mean": _mean(
                [r["tokens_per_sec"] for r in trial_results]
            ),
            "p50_token_ms_mean": _mean(
                [r["p50_token_ms"] for r in trial_results]
            ),
            "p90_token_ms_mean": _mean(
                [r["p90_token_ms"] for r in trial_results]
            ),
            "p99_token_ms_mean": _mean(
                [r["p99_token_ms"] for r in trial_results]
            ),
            "peak_mem_delta_mb": peak_mem_delta_mb,
        }
        config_results.append(agg)
        print(
            f"    tps={agg['tokens_per_sec_mean']:.1f} "
            f"ttft={agg['ttft_ms_mean']:.1f}ms "
            f"p50={agg['p50_token_ms_mean']:.2f}ms "
            f"peak_delta={agg['peak_mem_delta_mb']:.1f}MB"
        )

    payload: dict[str, Any] = {
        "release": "main26",
        "analysis": "generation_throughput",
        "model": model_id,
        "hardware": _get_hardware_info(),
        "context_tokens": int(input_ids.shape[1]),
        "decode_steps": n_decode,
        "warmup_repeats": warmup,
        "timed_repeats": repeats,
        "configs": config_results,
        "notes": [
            "Throughput measured as decode tokens/sec (excludes TTFT).",
            "Peak memory delta estimated via RSS; MPS unified memory is shared.",
            "Not a production benchmark; indicative only.",
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote generation throughput to {out_path}")
    return payload


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RFSN v10 Main 26 generation throughput benchmark"
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model ID",
    )
    parser.add_argument("--tokens", type=int, default=256, help="Context tokens")
    parser.add_argument("--decode", type=int, default=64, help="Decode steps")
    parser.add_argument("--repeats", type=int, default=5, help="Timed repeats")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup repeats")
    parser.add_argument(
        "--configs",
        default=(
            "baseline_fp16,k8_v3_gs64,k8_v4_gs64,k8_v5_gs64,"
            "k6_v6_gs64,k8_v4_gs32,k8_v5_gs32,k4_v4_gs64"
        ),
        help="Comma-separated config names",
    )
    parser.add_argument(
        "--out",
        default="artifacts/proof/main26/generation_throughput.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true",
        help="Trust remote code from HuggingFace",
    )
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    configs = [_parse_config(n) for n in config_names]

    _run_benchmark(
        model_id=args.model,
        tokens=args.tokens,
        n_decode=args.decode,
        repeats=args.repeats,
        warmup=args.warmup,
        configs=configs,
        device=device,
        out_path=Path(args.out),
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
