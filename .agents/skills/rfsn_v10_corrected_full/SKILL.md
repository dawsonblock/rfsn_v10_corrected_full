```markdown
# rfsn_v10_corrected_full Development Patterns

> Auto-generated skill from repository analysis

## Overview

This skill provides guidance on contributing to the `rfsn_v10_corrected_full` Python codebase. It covers the project's coding conventions, artifact and proof management workflows, and testing patterns. The repository is focused on generating, validating, and packaging release artifacts, including benchmarks and proof summaries, with a strong emphasis on consistency and integrity.

## Coding Conventions

- **File Naming:**  
  Use `snake_case` for all file and directory names.  
  _Example:_  
  ```
  check_release_integrity.py
  kernel_benchmark.json
  proof_summary.md
  ```

- **Import Style:**  
  Use relative imports within modules.  
  _Example:_  
  ```python
  from .utils import load_json
  ```

- **Export Style:**  
  Use named exports (explicit function/class definitions, no `__all__` or wildcard exports).  
  _Example:_  
  ```python
  def check_integrity(...):
      ...
  ```

- **Commit Messages:**  
  Freeform, no enforced prefix, average length ~66 characters.

## Workflows

### Release Artifact Generation and Proof Consistency

**Trigger:** When preparing a new release or updating benchmarks/proofs for a new 'Main' version.  
**Command:** `/new-release-artifacts`

1. **Rebrand or create new artifact directories/files**  
   - Create a new directory for the release under `artifacts/proof/mainXX/` (replace `XX` with the version).
   - Example:
     ```
     mkdir -p artifacts/proof/main12/
     ```

2. **Regenerate or update benchmark JSON files**  
   - Update or create:
     - `kernel_benchmark.json`
     - `fused_kernel_benchmark.json`
     - `generation_smoke.json`
     - `generation_throughput.json`
   - Place them in the new `mainXX` directory.

3. **Update or auto-generate proof summaries**  
   - Use scripts to generate `proof_summary.md` and `summary.json` from the JSON artifacts.
   - Example:
     ```bash
     python scripts/generate_proof_summary.py artifacts/proof/main12/
     ```

4. **Update README.md**  
   - Reflect new status, results, and any relevant changes for the new release.

5. **Update or harden integrity and consistency checking scripts**  
   - Ensure scripts like `check_release_integrity.py` and `check_proof_summary_consistency.py` are up to date.
   - Example:
     ```bash
     python scripts/check_release_integrity.py artifacts/proof/main12/
     python scripts/check_proof_summary_consistency.py artifacts/proof/main12/
     ```

6. **Run all integrity checks and package the release**  
   - Ensure all checks pass before finalizing the release package.

**Files Involved:**
- `artifacts/proof/main*/fused_kernel_benchmark.json`
- `artifacts/proof/main*/kernel_benchmark.json`
- `artifacts/proof/main*/generation_smoke.json`
- `artifacts/proof/main*/generation_throughput.json`
- `artifacts/proof/main*/proof_summary.md`
- `artifacts/proof/main*/summary.json`
- `README.md`
- `scripts/check_release_integrity.py`
- `scripts/check_proof_summary_consistency.py`
- `scripts/generate_proof_summary.py`

**Frequency:** ~1-2 times per month

---

## Testing Patterns

- **Framework:** Unknown (no standard framework detected).
- **Test File Pattern:** Test files are named with `.test.` in the filename.
  - _Example:_ `utils.test.py`
- **Location:** Test files are typically placed alongside the code they test.
- **Running Tests:**  
  Since the framework is unknown, run test files directly with Python:
  ```bash
  python path/to/file.test.py
  ```

## Commands

| Command                | Purpose                                                                 |
|------------------------|-------------------------------------------------------------------------|
| /new-release-artifacts | Generate new release artifacts, update proofs, and run consistency checks|
```
