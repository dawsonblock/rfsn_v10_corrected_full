# Fused Kernel Plan (Main14+)

## Status
This document is preparation-only for a future single-kernel implementation.
Main13 ships a multi-kernel Metal route and does not claim fused-kernel completion.

## Target Operation
Single Metal kernel pipeline:
1. packed uint32 decode
2. symmetric dequant
3. WHT64 transform
4. incoherent sign application
5. output tensor write

## Input Contract
- packed: uint32 flat buffer containing grouped packed codes
- scales: float16/float32 grouped scale tensor
- n_values: total scalar values to reconstruct
- shape: output tensor shape [B, H, T, D]
- bits: quant bits per code (current target: 8 and 3 compatibility)
- group_size: quant group size used by packer
- seed: deterministic sign hash seed

## Output Contract
- output tensor with shape [B, H, T, D]
- dtype: float16 or float32
- deterministic for same packed/scales/seed input

## Quant Code Format
- symmetric quantization centered on qmax
- packed as contiguous uint32 words
- code extraction uses bit offset by bits and group_size

## Scale Format
- grouped scale per quant group
- scale index = value_index // group_size
- supports float16 storage, compute in float32

## WHT Block Layout
- last-dimension block size: 64
- D must be divisible by 64
- WHT ordering must match reference implementation
- normalization must preserve reference-equivalence thresholds

## Sign Hash Convention
- deterministic +/-1 sign vector from seed
- sign stream shape-compatible with reconstructed tensor
- convention must match existing runtime/reference route

## Threadgroup Layout (Target)
- map one threadgroup to a fixed chunk of output values
- prefer coalesced reads for packed words and grouped scales
- avoid host roundtrips between dequant, WHT, and sign stages

## Validation Tests (Required)
- parity vs sequential_reference across required shapes
- WHT self-inverse checks on Metal path
- strict invalid-code rejection parity with reference path
- no fallback accepted in strict mode tests
- route label must reflect fused path only when truly single-kernel

## Benchmark Gates (Required)
- benchmark rows must record fallback_used, cosine_vs_reference, max_abs_diff_vs_reference
- valid row policy:
  - fallback_used == false
  - cosine_vs_reference >= 0.999
  - max_abs_diff_vs_reference <= 1e-3
- speedup claims allowed only for valid rows

## Non-Goals for Main13
- No fused kernel code path shipped.
- No fused-kernel performance claim in README/proof summary.
