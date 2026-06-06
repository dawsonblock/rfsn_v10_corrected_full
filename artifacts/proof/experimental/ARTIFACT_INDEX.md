# Experimental Proof Artifact Index

**Release:** experimental  
**Stable Default:** `k8_v5_gs64`  
**QJL Status:** failed, disabled  
**Promoted to Default:** false  
**Experimental Status:** research_only  
**Teacher-Forced Baseline Status:** valid (identity guard confirmed)  
**Promotion Blocked By:** teacher_forced_logit_drift, decode_update_failure, qjl_failed  
**Decode-Update Diagnosis:** stable configs pass decode-update; experimental configs fail

## Artifact Manifest

| Artifact | File | Status | Rows | Purpose |
|----------|------|--------|------|---------|
| comparison | `comparison_summary.json` | executed | — | 0.5B model quality comparison across configs |
| memory | `memory_accounting.json` | executed | — | Per-config compressed memory with real-model basis |
| throughput | `throughput.json` | executed | — | Synthetic KV throughput benchmark |
| real_generation_throughput | `real_generation_throughput.json` | executed | — | Teacher-forced + free-running generation benchmark |
| decode_update_trace | `decode_update_trace.json` | **executed** | **70** | Step-by-step decode path quality trace |
| decode_append_kv_diff | `decode_append_kv_diff.json` | **executed** | **10** | Old-cache preservation and new-token K/V append diagnostics |
| kv_roundtrip_by_context | `kv_roundtrip_by_context.json` | executed | — | Direct KV quantizer roundtrip by prompt length |
| prefill_decode_split | `prefill_decode_split.json` | executed | — | Prefill-vs-decode isolation (A/B/C/D modes) |
| short_prompt_drift_trace | `short_prompt_drift_trace.json` | executed | — | Step-by-step short-prompt logit drift trace |
| qjl | `qjl_attention_score.json` | executed | — | QJL benchmark result (failed) |
| layer_policy | `layer_policy.json` | executed | — | Conservative per-layer policy |
| qwen_1_5b | `qwen_1_5b/` | executed | — | 1.5B model validation directory |

## Decode Diagnostics Summary

### decode_update_trace.json (70 rows)

Step-level decode comparison: FP16 vs quantized, per config × prompt_tokens × decode_step.

| Config | Result |
|--------|--------|
| k8_v5_gs64 | All steps pass (cosine ≥ 0.999) |
| k8_v5_gs32 | All steps pass (cosine ≥ 0.999) |
| turbo_polar | All steps degraded (cosine 0.65–0.99, top5 0.2–0.8) |
| adaptive | All steps degraded (cosine 0.67–0.96, top5 0.4–0.8) |
| experimental_hybrid | All steps degraded (cosine 0.76–0.99, top5 0.2–1.0) |

### decode_append_kv_diff.json (10 rows)

Pre/post-append K/V analysis: old-cache preservation vs new-token quantization error.

| Config | old_k_cos | new_k_cos | cache_ok |
|--------|-----------|-----------|----------|
| k8_v5_gs64 | ≥ 0.9999 | ≥ 0.9994 | true |
| k8_v5_gs32 | ≥ 0.9999 | ≥ 0.9996 | true |
| turbo_polar | ≥ 0.9983 | ≥ 0.9573 | true |
| adaptive | ≥ 0.9983 | ≥ 0.9326 | true |
| experimental_hybrid | ≥ 0.9960 | ≥ 0.9443 | true |

**Finding:** Old-cache is not corrupted by appending. New-token K/V quantization error is the primary source of degradation in experimental configs. Stable configs have high new-token quality.

## Classification Rules

- No experimental mode may be classified as a candidate without real-generation data.
- Teacher-forced baseline must be exact identity (cosine=1.0, top5=1.0, KL=0.0).
- If baseline identity fails, all configs are blocked as `needs_valid_teacher_forced_baseline`.
- Teacher-forced real-generation failure is a hard reject (`rejected_generation_quality`).
- Free-running divergence without teacher-forced failure is `generation_divergence_observed`.
- No config using raw `uint32` fallback (>8-bit) may be called memory-optimized.
- QJL remains disabled until its attention score benchmark passes.
- The default runtime mode is locked to `k8_v5_gs64`.

## Current Config Status

| Config | Classification |
|--------|---------------|
| baseline_fp16 | reference |
| k8_v5_gs64 | stable default, locked, teacher-forced drift under investigation |
| k8_v5_gs32 | quality candidate, teacher-forced drift under investigation |
| experimental_hybrid | free-running match, teacher-forced failed |
| turbo_polar | rejected generation divergence |
| adaptive | rejected generation divergence |
| QJL | disabled |
