#!/usr/bin/env python3
"""Main12 real-model KV validation runner.

Runs a local Hugging Face causal LM in three modes:
- fp16_baseline
- compressed_kv
- compressed_kv_sparse

The compressed modes use RFSN TurboQuant KV compression over model past
key/value tensors, then decode a held-out token and compare logits.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import mlx.core as mx
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from rfsn_v10.kv_manager import RFSNTurboQuantKVManager


def _not_run_payload(reason: str) -> dict[str, Any]:
    return {
        "status": "not_run",
        "reason": reason,
        "mode_results": [],
    }


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.reshape(-1).float()
    b_f = b.reshape(-1).float()
    return float(F.cosine_similarity(a_f, b_f, dim=0).item())


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


def _block_select_indices(
    seq_len: int,
    top_k_ratio: float,
    block_size: int = 64,
    reserved_sink_blocks: int = 1,
    reserved_recent_blocks: int = 2,
) -> torch.Tensor:
    num_blocks = max(1, math.ceil(seq_len / block_size))
    k_active = max(1, math.ceil(num_blocks * top_k_ratio))

    reserved = []
    for idx in range(min(reserved_sink_blocks, num_blocks)):
        reserved.append(idx)
    for offset in range(reserved_recent_blocks):
        idx = num_blocks - 1 - offset
        if idx >= 0 and idx not in reserved:
            reserved.append(idx)

    selected = []
    for idx in reserved:
        if idx not in selected:
            selected.append(idx)

    for idx in range(num_blocks - 1, -1, -1):
        if len(selected) >= k_active:
            break
        if idx not in selected:
            selected.append(idx)

    selected = sorted(set(selected))
    token_indices = []
    for b in selected:
        start = b * block_size
        end = min(seq_len, (b + 1) * block_size)
        token_indices.extend(range(start, end))

    if not token_indices:
        token_indices = [seq_len - 1]

    return torch.tensor(token_indices, dtype=torch.long)


def _compress_decompress_past(
    past_key_values,
    *,
    sparse: bool,
    top_k_ratio: float,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    rebuilt: list[tuple[torch.Tensor, torch.Tensor]] = []

    for layer_idx, (k_t, v_t) in enumerate(past_key_values):
        k_np = k_t.detach().to("cpu", dtype=torch.float32).numpy()
        v_np = v_t.detach().to("cpu", dtype=torch.float32).numpy()

        bsz, heads, seq, dim = k_np.shape
        dim_padded = int(math.ceil(dim / 64.0) * 64)
        if dim_padded != dim:
            pad = dim_padded - dim
            k_np = __import__("numpy").pad(k_np, ((0, 0), (0, 0), (0, 0), (0, pad)))
            v_np = __import__("numpy").pad(v_np, ((0, 0), (0, 0), (0, 0), (0, pad)))

        k_mx = mx.array(k_np)
        v_mx = mx.array(v_np)

        mgr = RFSNTurboQuantKVManager(
            k_bits=8,
            v_bits=3,
            use_wht=True,
            use_incoherent_signs=True,
            prefer_metal_kernels=True,
            strict_metal=True,
            max_memory_gb=2.0,
        )

        key = f"layer_{layer_idx}"
        mgr.store(key, k_mx, v_mx, token_count=seq)
        rec = mgr.retrieve(key, out_dtype=mx.float32)
        if rec is None:
            raise RuntimeError("Unexpected cache miss during validation")
        rk_mx, rv_mx = rec
        rk = torch.tensor(rk_mx.tolist(), dtype=torch.float32)
        rv = torch.tensor(rv_mx.tolist(), dtype=torch.float32)

        rk = rk[..., :dim]
        rv = rv[..., :dim]

        if sparse:
            idx = _block_select_indices(seq, top_k_ratio=top_k_ratio)
            rk = rk.index_select(2, idx)
            rv = rv.index_select(2, idx)

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


def _evaluate_mode(
    mode: str,
    *,
    model,
    context_ids: torch.Tensor,
    decode_token: torch.Tensor,
    baseline_logits: torch.Tensor,
    baseline_ppl: float,
    device: torch.device,
) -> dict[str, Any]:
    with torch.no_grad():
        context_out = model(input_ids=context_ids, use_cache=True)
    past_raw = context_out.past_key_values
    past_legacy, cache_cls = _to_legacy_cache(past_raw)

    if mode == "compressed_kv":
        past_legacy = _compress_decompress_past(
            past_legacy,
            sparse=False,
            top_k_ratio=1.0,
            device=device,
        )
    elif mode == "compressed_kv_sparse":
        past_legacy = _compress_decompress_past(
            past_legacy,
            sparse=True,
            top_k_ratio=0.5,
            device=device,
        )

    past = _from_legacy_cache(past_legacy, cache_cls)

    with torch.no_grad():
        out = model(input_ids=decode_token, past_key_values=past, use_cache=True)
    logits = out.logits[:, -1, :]

    target_id = int(decode_token[0, 0].item())
    ppl = _perplexity_for_target(logits, target_id)

    top1_match = float(torch.argmax(logits, dim=-1).item() == torch.argmax(baseline_logits, dim=-1).item())
    result = {
        "mode": mode,
        "tokens_tested": int(context_ids.shape[1]) + 1,
        "logit_cosine": _cosine(logits, baseline_logits),
        "logit_max_abs_diff": float(torch.max(torch.abs(logits - baseline_logits)).item()),
        "top1_token_match_rate": top1_match,
        "top5_overlap": _topk_overlap(logits, baseline_logits, k=5),
        "perplexity_delta": float(ppl - baseline_ppl),
        "generated_text_diff_summary": "compared against fp16_baseline next-token logits",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate KV quality against a real local model")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--out", default="artifacts/proof/main12/real_model_validation.json")
    parser.add_argument("--max-input-tokens", type=int, default=128)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    prompt_path = Path(args.prompt_file)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        payload = _not_run_payload("model_path_missing")
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Real model validation not run: {payload['reason']}")
        return

    if not prompt_path.exists():
        payload = _not_run_payload("prompt_file_missing")
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Real model validation not run: {payload['reason']}")
        return

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device.type == "mps" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_path.as_posix(), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path.as_posix(),
        local_files_only=True,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()

    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        payload = _not_run_payload("prompt_file_empty")
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Real model validation not run: {payload['reason']}")
        return

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"][:, : args.max_input_tokens]
    if input_ids.shape[1] < 2:
        payload = _not_run_payload("prompt_too_short")
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Real model validation not run: {payload['reason']}")
        return

    input_ids = input_ids.to(device)
    context_ids = input_ids[:, :-1]
    decode_token = input_ids[:, -1:]

    with torch.no_grad():
        baseline_ctx = model(input_ids=context_ids, use_cache=True)
        baseline_out = model(input_ids=decode_token, past_key_values=baseline_ctx.past_key_values, use_cache=True)

    baseline_logits = baseline_out.logits[:, -1, :]
    baseline_target = int(decode_token[0, 0].item())
    baseline_ppl = _perplexity_for_target(baseline_logits, baseline_target)

    results = [
        {
            "mode": "fp16_baseline",
            "tokens_tested": int(input_ids.shape[1]),
            "logit_cosine": 1.0,
            "logit_max_abs_diff": 0.0,
            "top1_token_match_rate": 1.0,
            "top5_overlap": 1.0,
            "perplexity_delta": 0.0,
            "generated_text_diff_summary": "baseline reference",
        }
    ]

    results.append(
        _evaluate_mode(
            "compressed_kv",
            model=model,
            context_ids=context_ids,
            decode_token=decode_token,
            baseline_logits=baseline_logits,
            baseline_ppl=baseline_ppl,
            device=device,
        )
    )
    results.append(
        _evaluate_mode(
            "compressed_kv_sparse",
            model=model,
            context_ids=context_ids,
            decode_token=decode_token,
            baseline_logits=baseline_logits,
            baseline_ppl=baseline_ppl,
            device=device,
        )
    )

    with torch.no_grad():
        gen = model.generate(
            input_ids=input_ids,
            max_new_tokens=32,
            do_sample=False,
            use_cache=True,
        )
    generated = tokenizer.decode(gen[0], skip_special_tokens=True)

    payload: dict[str, Any] = {
        "status": "run",
        "model_path": model_path.as_posix(),
        "device": str(device),
        "mode_results": results,
        "notes": [
            "Actual local model inference executed for baseline, compressed_kv, and compressed_kv_sparse modes.",
            "compressed_kv_sparse mode currently uses deterministic block pruning proxy over reconstructed KV tensors.",
        ],
        "generated_sample": generated,
    }

    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote real model validation to {out_path}")
    print(f"Device: {device}")
    for row in results:
        print(
            row["mode"],
            f"cos={row['logit_cosine']:.6f}",
            f"max_abs={row['logit_max_abs_diff']:.6f}",
            f"top1={row['top1_token_match_rate']:.3f}",
            f"top5={row['top5_overlap']:.3f}",
            f"ppl_delta={row['perplexity_delta']:.6f}",
        )


if __name__ == "__main__":
    main()
