# Experimental Proof Artifact Index

**Release:** experimental  
**Stable Default:** `k8_v5_gs64`  
**QJL Status:** failed, disabled  
**Promoted to Default:** false

## Artifact Manifest

| Artifact | File | Purpose |
|----------|------|---------|
| comparison | `comparison_summary.json` | 0.5B model quality comparison across configs |
| memory | `memory_accounting.json` | Per-config compressed memory with real-model basis |
| throughput | `throughput.json` | Synthetic KV throughput benchmark |
| real_generation_throughput | `real_generation_throughput.json` | End-to-end greedy decode throughput with compressed KV |
| qjl | `qjl_attention_score.json` | QJL benchmark result (failed) |
| layer_policy | `layer_policy.json` | Conservative per-layer policy |
| qwen_1_5b | `qwen_1_5b/` | 1.5B model validation directory |

## Classification Rules

- No experimental mode may be classified as a candidate without throughput data.
- No config using raw `uint32` fallback (>8-bit) may be called memory-optimized.
- QJL remains disabled until its attention score benchmark passes.
- The default runtime mode is locked to `k8_v5_gs64`.

## Candidate Roles

- `k8_v5_gs64`: production-facing stable baseline (default)
- `k8_v5_gs32`: conservative high-quality / layer-policy candidate
- `turbo_polar`: experimental speed study target
- `adaptive`: experimental quality study target
- `experimental_hybrid`: compression study target
