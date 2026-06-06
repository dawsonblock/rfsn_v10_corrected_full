# Experimental Proof Artifact Index

**Release:** experimental  
**Stable Default:** `k8_v5_gs64`  
**QJL Status:** failed, disabled  
**Promoted to Default:** false  
**Experimental Status:** research_only  
**Teacher-Forced Baseline Status:** valid (identity guard confirmed)  
**Promotion Blocked By:** decode_quantization_weakness, throughput_overhead, qjl_failed

## Artifact Manifest

| Artifact | File | Purpose |
|----------|------|---------|
| comparison | `comparison_summary.json` | 0.5B model quality comparison across configs |
| memory | `memory_accounting.json` | Per-config compressed memory with real-model basis |
| throughput | `throughput.json` | Synthetic KV throughput benchmark |
| real_generation_throughput | `real_generation_throughput.json` | Teacher-forced + free-running generation benchmark |
| decode_update_trace | `decode_update_trace.json` | Step-by-step decode path quality trace |
| decode_append_kv_diff | `decode_append_kv_diff.json` | Pre/post-append KV tensor comparison |
| kv_roundtrip_by_context | `kv_roundtrip_by_context.json` | Direct KV quantizer roundtrip by prompt length |
| prefill_decode_split | `prefill_decode_split.json` | Prefill-vs-decode isolation (A/B/C/D modes) |
| short_prompt_drift_trace | `short_prompt_drift_trace.json` | Step-by-step short-prompt logit drift trace |
| qjl | `qjl_attention_score.json` | QJL benchmark result (failed) |
| layer_policy | `layer_policy.json` | Conservative per-layer policy |
| qwen_1_5b | `qwen_1_5b/` | 1.5B model validation directory |

## Classification Rules

- No experimental mode may be classified as a candidate without real-generation data.
- Teacher-forced baseline must be exact identity (cosine=1.0, top5=1.0, KL=0.0).
- If baseline identity fails, all configs are blocked as `needs_valid_teacher_forced_baseline`.
- Teacher-forced real-generation failure is a hard reject (`rejected_generation_quality`).
- Free-running divergence without teacher-forced failure is `generation_divergence_observed`.
- No config using raw `uint32` fallback (>8-bit) may be called memory-optimized.
- QJL remains disabled until its attention score benchmark passes.
- The default runtime mode is locked to `k8_v5_gs64`.

## Candidate Roles

- `k8_v5_gs64`: stable default, short-prompt drift under investigation
- `k8_v5_gs32`: quality candidate, short-prompt drift under investigation
- `turbo_polar`: rejected_generation_quality
- `adaptive`: rejected_generation_quality
- `experimental_hybrid`: rejected_generation_quality
