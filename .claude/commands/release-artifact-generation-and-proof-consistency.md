---
name: release-artifact-generation-and-proof-consistency
description: Workflow command scaffold for release-artifact-generation-and-proof-consistency in rfsn_v10_corrected_full.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /release-artifact-generation-and-proof-consistency

Use this workflow when working on **release-artifact-generation-and-proof-consistency** in `rfsn_v10_corrected_full`.

## Goal

Generates new release artifacts for a new 'Main' version, updates proof summaries, benchmarks, and ensures consistency and integrity checks across all artifacts and documentation.

## Common Files

- `artifacts/proof/main*/fused_kernel_benchmark.json`
- `artifacts/proof/main*/kernel_benchmark.json`
- `artifacts/proof/main*/generation_smoke.json`
- `artifacts/proof/main*/generation_throughput.json`
- `artifacts/proof/main*/proof_summary.md`
- `artifacts/proof/main*/summary.json`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Rebrand or create new artifact directories/files for the new Main version (e.g., artifacts/proof/mainXX/...)
- Regenerate or update benchmark JSON files (e.g., kernel_benchmark.json, fused_kernel_benchmark.json, generation_smoke.json, generation_throughput.json)
- Update or auto-generate proof_summary.md and summary.json from JSON artifacts
- Update README.md to reflect new status and results
- Update or harden integrity and consistency checking scripts (e.g., check_release_integrity.py, check_proof_summary_consistency.py, generate_proof_summary.py)

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.