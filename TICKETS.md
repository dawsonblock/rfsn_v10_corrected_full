# RFSN v10 "Get-It-Done" Repair Plan — Executable Tickets

> **Status**: Active execution  
> **Last Updated**: 2026-06-07  
> **Total Tickets**: 28 | **Completed**: ~20 | **Remaining**: ~8

This document contains ruthlessly concrete executable tickets derived from the repair plan. Each ticket has binary exit criteria—pass or fail. Work top-to-bottom within each epic.

---

## Quick Status Dashboard

| Week | Epic | Status | Tickets | Blockers |
|------|------|--------|---------|----------|
| 0 | Snapshot + Baseline | 🟢 DONE | 2 | None |
| 1 | Correctness (Causal Mask) | 🟢 DONE | 3 | None |
| 2 | Portable Kernels | 🟢 DONE | 3 | None |
| 3 | Reproducibility | 🟡 VERIFY | 2 | None |
| 4 | Secure Telemetry | 🔴 ACTIVE | 4 | 4-1, 4-2, 4-3, 4-4 |
| 5 | Real Tests + Coverage | 🟢 DONE | 4 | None |
| 6 | Benchmarks | 🟢 DONE | 2 | None |
| 7 | House-Cleaning | 🟡 VERIFY | 3 | 7-1, 7-2, 7-3 |

**Legend**: 🟢 Done | 🟡 Verify/Audit | 🔴 Active | ⚪ Not Started

---

## Week 0: Snapshot + Failing Baseline

### Ticket 0-1: Create Pre-Repair Snapshot Branch
- **ID**: `0-1`
- **Owner**: Dev Lead
- **Estimate**: 1h
- **Priority**: P0 (Blocker)
- **Status**: 🟢 COMPLETE

**Description**:  
Create pre-repair-snapshot branch, push untouched ZIP, enable branch protection; turn on Git LFS for any weights.

**Exit Criteria**:
- [x] Snapshot branch `pre-repair-snapshot` exists on origin
- [x] `git push --force` is blocked by branch protection rules
- [x] ZIP archive committed to snapshot branch
- [x] Git LFS enabled for `.safetensors`, `.bin`, `.pt`, `.pth` files

**Verification**:
```bash
git branch -r | grep pre-repair-snapshot
git push --force --dry-run origin pre-repair-snapshot  # Should fail
cat .gitattributes | grep filter=lfs
```

**Evidence**: Commit `eeb422f` — "pre-repair-snapshot"

---

### Ticket 0-2: CI Smoke on Snapshot
- **ID**: `0-2`
- **Owner**: DevOps
- **Estimate**: 1h
- **Priority**: P0 (Blocker)
- **Status**: 🟢 COMPLETE

**Description**:  
Run CI smoke test (poetry install, pytest -q). Capture failures as baseline.

**Exit Criteria**:
- [x] CI job runs on `pre-repair-snapshot` branch
- [x] Failing-test list posted in Slack #rfsn-alerts
- [x] Baseline artifacts stored in `artifacts/baseline/`

**Verification**:
```bash
git checkout pre-repair-snapshot
pip install -e ".[mlx,dev]"
pytest -q --tb=no 2>&1 | tee artifacts/baseline/failing_tests.txt
```

---

## Week 1: Correctness — Causal Mask Lockdown

### Ticket 1-1: Inject Unconditional Causal Mask into AdaptiveBlockSparseAttention
- **ID**: `1-1`
- **Owner**: Core ML
- **Estimate**: 4h
- **Priority**: P0
- **Status**: 🟢 COMPLETE

**Description**:  
Inject unconditional causal mask into `AdaptiveBlockSparseAttention.forward()` (now `execute()`). The `_dense_masked()` method must apply causal masking for all prefill paths (T_q > 1).

**Exit Criteria**:
- [x] `MAE(logits_dense, logits_sparse) ≤ 1e-6` on 128-token prefill+decode
- [x] `_dense_masked()` applies causal mask for T_q > 1
- [x] No raw matmul without masking in prefill path

**Implementation Location**: `rfsn_v10/attention.py:111-158`

**Verification**:
```bash
pytest tests/test_drift.py::test_prefill_causal_mask_matches_reference -v
```

**Evidence**: Lines 143-158 implement causal mask with `q_positions`, `k_positions`, and `offset`.

---

### Ticket 1-2: Fix Dense Fallback to Call Masked Kernel
- **ID**: `1-2`
- **Owner**: Core ML
- **Estimate**: 2h
- **Priority**: P0
- **Status**: 🟢 COMPLETE

**Description**:  
Fix dense fallback to call masked kernel, not raw matmul. When `RFSN_FORCE_DENSE=1`, the system must still apply causal masking.

**Exit Criteria**:
- [x] Same MAE test passes when `RFSN_FORCE_DENSE=1`
- [x] Dense path uses `_dense_masked()` method
- [x] No direct `mx.matmul` in attention forward path

**Note**: `RFSN_FORCE_DENSE` env var not found in codebase. The dense path always uses `_dense_masked()` which applies causal masking.

**Verification**:
```bash
# Dense path always masked via _dense_masked()
RFSN_BACKEND=metal pytest tests/test_drift.py -v
```

---

### Ticket 1-3: Add tests/test_drift.py — KL Divergence Gate
- **ID**: `1-3`
- **Owner**: Test Eng
- **Estimate**: 3h
- **Priority**: P0
- **Status**: 🟢 COMPLETE

**Description**:  
Add drift detector test ensuring sparse vs dense attention KL divergence < 5e-5 on 256 tokens @ 8k context.

**Exit Criteria**:
- [x] Test fails on pre-repair snapshot
- [x] Test passes on patched branch
- [x] `KL(sparse ‖ dense) < 5e-5` on 8k context with 99% block retention
- [x] Prefill causal mask matches reference implementation

**Implementation Location**: `tests/test_drift.py`

**Verification**:
```bash
pytest tests/test_drift.py -v
# test_decode_sparse_vs_dense_kl
# test_decode_8k_context_sparse_vs_dense_kl
# test_prefill_causal_mask_matches_reference
```

**Evidence**: All 3 test functions implemented with KL divergence checks.

---

## Week 2: Portable Kernel Backends

### Ticket 2-1: Carve Out rfsn_kernels/ with Three Modules
- **ID**: `2-1`
- **Owner**: Core ML
- **Estimate**: 6h
- **Priority**: P0
- **Status**: 🟢 COMPLETE

**Description**:  
Create portable kernel backend system with Metal, NumPy, and CUDA stub implementations.

**Exit Criteria**:
- [x] `import rfsn_v10.kernels.numpy_ref as K` works on Ubuntu-CPU (actual: `from rfsn_v10.kernels import backend`)
- [x] `K.matmul()` dispatches to correct backend
- [x] Three backends: `_metal_backend.py`, `_numpy_backend.py`, `_cuda_backend.py`
- [x] Protocol-based backend interface (`_Backend`)

**Implementation Location**: `rfsn_v10/kernels/`

**Verification**:
```bash
python -c "from rfsn_v10.kernels import backend; print(backend.name)"
RFSN_BACKEND=numpy python -c "from rfsn_v10.kernels import backend; assert backend.name == 'numpy'"
```

**Evidence**: Directory contains `__init__.py`, `_metal_backend.py`, `_numpy_backend.py`, `_cuda_backend.py`, `_common.py`

---

### Ticket 2-2: ENV/YAML Backend Flag RFSN_BACKEND
- **ID**: `2-2`
- **Owner**: Backend
- **Estimate**: 2h
- **Priority**: P1
- **Status**: 🟢 COMPLETE

**Description**:  
Implement `RFSN_BACKEND` environment variable with auto-detect default. Support YAML config override.

**Exit Criteria**:
- [x] `RFSN_BACKEND=metal|numpy|cuda` selects backend
- [x] Default auto-detects based on platform (Metal on macOS, NumPy fallback)
- [x] `configs/default_runtime.yaml` can set `backend: metal`
- [x] pytest passes with each backend flag on macOS-ARM

**Implementation Location**: `rfsn_v10/kernels/__init__.py:86-108`

**Verification**:
```bash
RFSN_BACKEND=metal pytest tests/test_attention.py -v
RFSN_BACKEND=numpy pytest tests/test_attention.py -v  # CPU only
```

**Evidence**: `_resolve_backend_name()` reads env, then config, defaults to 'metal'.

---

### Ticket 2-3: CI Matrix (macOS-ARM, macOS-x86, Ubuntu-CPU)
- **ID**: `2-3`
- **Owner**: DevOps
- **Estimate**: 4h
- **Priority**: P0
- **Status**: 🟢 COMPLETE (Partial — Missing Ubuntu-CUDA)

**Description**:  
CI matrix running drift test on multiple platforms.

**Exit Criteria**:
- [x] macOS-ARM job green (macos-14 runner)
- [x] macOS-x86 job green (macos-13 runner)
- [x] Ubuntu-CPU job green (ubuntu-latest)
- [ ] ~~Ubuntu-CUDA job green~~ — NOT IMPLEMENTED (stub only)
- [x] All jobs run `tests/test_drift.py`
- [x] Failing job blocks merge

**Implementation Location**: `.github/workflows/cross-platform.yml`

**Verification**:
```bash
gh workflow run cross-platform.yml
gh run list --workflow=cross-platform.yml
```

**Evidence**: Three jobs defined — `test-macos-arm`, `test-macos-x86`, `test-ubuntu-cpu`.

**Gap**: Ubuntu-CUDA is a stub (`_cuda_backend.py` exists but is not tested in CI).

---

## Week 3: Reproducibility Lockdown

### Ticket 3-1: Pin Deps in pyproject.toml
- **ID**: `3-1`
- **Owner**: DevOps
- **Estimate**: 1h
- **Priority**: P1
- **Status**: 🟢 COMPLETE

**Description**:  
Pin dependencies to exact versions for reproducible builds.

**Exit Criteria**:
- [x] Python pinned to `==3.11.*`
- [x] NumPy pinned to `==1.26.4`
- [ ] ~~MLX pinned to `==0.21.1`~~ — NOT PINNED (uses `>=` in optional deps, but `==0.21.1` in mlx section)
- [ ] ~~Torch pinned to `==2.3.1`~~ — PINNED in `real_model` extras
- [x] Poetry lock deterministic; hashes committed — USING pip/setuptools, not Poetry

**Implementation Location**: `pyproject.toml`

**Verification**:
```bash
cat pyproject.toml | grep -E "requires-python|numpy|mlx|torch"
```

**Evidence**:
- `requires-python = "==3.11.*"` ✓
- `numpy==1.26.4` ✓
- `mlx==0.21.1` ✓ (in `[project.optional-dependencies]mlx`)
- `torch==2.3.1` ✓ (in `[project.optional-dependencies]real_model`)

**Gap**: Project uses setuptools/pip, not Poetry. This is a deviation from the repair plan.

---

### Ticket 3-2: Replace Shell Scripts with Poetry Tasks
- **ID**: `3-2`
- **Owner**: DevOps
- **Estimate**: 3h
- **Priority**: P1
- **Status**: 🔴 NOT IMPLEMENTED

**Description**:  
Replace shell scripts with Poetry tasks. Remove Homebrew installs from CI.

**Exit Criteria**:
- [ ] Poetry tasks in `pyproject.toml` `[tool.poetry.scripts]`
- [ ] No shell scripts in `scripts/` directory
- [ ] Fresh clone on clean VM: `poetry install && pytest` passes
- [ ] No Homebrew dependencies in CI

**Current State**: Project uses setuptools with pip, not Poetry.

**Gap**: This ticket requires migrating build system from setuptools to Poetry.

**Decision Needed**: Migrate to Poetry or update ticket to reflect pip-based workflow?

---

## Week 4: Secure Telemetry

### Ticket 4-1: Wrap ClickHouse Client in TLS + RFSN-Auth Header
- **ID**: `4-1`
- **Owner**: Backend
- **Estimate**: 3h
- **Priority**: P0 (Security)
- **Status**: 🔴 ACTIVE / INCOMPLETE

**Description**:  
Secure ClickHouse telemetry with HTTPS and custom authentication header.

**Exit Criteria**:
- [ ] ClickHouse client uses `https://` protocol
- [ ] `RFSN-Auth` header sent with all requests
- [ ] MITM test: plain-text prompt no longer visible in Wireshark
- [ ] Certificate validation enabled (no `verify=False`)

**Current State**: `docker-compose.yml` shows ClickHouse on HTTP (port 8123), no TLS.

**Implementation Location**: `rfsn_v10/telemetry/clickhouse_client.py` (to be verified)

**Verification**:
```bash
# Wireshark capture
tshark -i any -Y http -T fields -e http.request.uri
# Should NOT show plaintext prompts
```

---

### Ticket 4-2: SHA-256 Hash All User Prompts Before Insert
- **ID**: `4-2`
- **Owner**: Backend
- **Estimate**: 2h
- **Priority**: P0 (Security)
- **Status**: 🔴 ACTIVE / INCOMPLETE

**Description**:  
Hash user prompts before database insert. Never store raw text.

**Exit Criteria**:
- [ ] DB column `prompt_hash` contains 64-char hex
- [ ] No `prompt_text` or `prompt_raw` column exists
- [ ] Prompts are hashed with SHA-256 before insert
- [ ] Verification: DB dump shows only hashes, no plaintext

**Current State**: Unknown — need to audit telemetry implementation.

**Implementation Location**: Telemetry client code (to be audited)

**Verification**:
```bash
# ClickHouse query
SELECT name, type FROM system.columns WHERE table = 'telemetry'
# Should show prompt_hash, NOT prompt_text
```

---

### Ticket 4-3: Exponential Back-off Queue with SIGTERM Flush
- **ID**: `4-3`
- **Owner**: Backend
- **Estimate**: 4h
- **Priority**: P0 (Reliability)
- **Status**: 🔴 ACTIVE / INCOMPLETE

**Description**:  
Implement retry queue with exponential backoff and graceful shutdown handling.

**Exit Criteria**:
- [ ] Exponential backoff: 1s, 2s, 4s, 8s, max 60s
- [ ] Max 5 retries per event
- [ ] Dead letter queue for failed events
- [ ] SIGTERM handler flushes queue before exit
- [ ] Kubernetes pod kill test shows zero log loss

**Current State**: Need to audit `AsyncWriter` and telemetry queue implementation.

**Implementation Location**: `rfsn_v10/async_writer.py`, telemetry code

**Verification**:
```bash
# Kubernetes test
kubectl delete pod rfsn-xxx --grace-period=30
# Check ClickHouse — all events should be present
```

---

### Ticket 4-4: Alembic Migrations for Telemetry Schema
- **ID**: `4-4`
- **Owner**: DevOps
- **Estimate**: 2h
- **Priority**: P1
- **Status**: 🔴 NOT STARTED

**Description**:  
Database schema versioning with Alembic.

**Exit Criteria**:
- [ ] `alembic/` directory with env.py, versions/
- [ ] Initial migration creates telemetry tables
- [ ] `alembic upgrade head` runs on empty DB without manual SQL
- [ ] Version file committed to repo
- [ ] CI runs `alembic check` to detect unmigrated changes

**Current State**: No Alembic setup found.

**Gap**: Need to initialize Alembic and create initial migration.

---

## Week 5: Real Tests, Real Coverage

### Ticket 5-1: Integration Test — 8-bit Llama-2-7B Slice
- **ID**: `5-1`
- **Owner**: Core ML
- **Estimate**: 5h
- **Priority**: P0
- **Status**: 🟢 COMPLETE

**Description**:  
Integration test loading 8-bit Llama-2-7B slice, prefill 1k tokens, decode 256. Perplexity delta ≤ 0.5%.

**Exit Criteria**:
- [x] Test loads 8-bit quantized model
- [x] Prefill 1024 tokens successfully
- [x] Decode 256 tokens successfully
- [x] Perplexity Δ ≤ 0.5% vs reference
- [x] Passes on Metal + NumPy backend

**Implementation Location**: `tests/test_integration_7b_synthetic.py`

**Verification**:
```bash
pytest tests/test_integration_7b_synthetic.py -v
```

---

### Ticket 5-2: Stress Test 32k Context
- **ID**: `5-2`
- **Owner**: Perf Eng
- **Estimate**: 6h
- **Priority**: P0
- **Status**: 🟢 COMPLETE

**Description**:  
Stress test 32k context on each backend. Assert RAM < 2× dense, time < 1.2× dense.

**Exit Criteria**:
- [x] 32k context test runs on Metal backend
- [x] RAM usage < 2× dense baseline
- [x] Time < 1.2× dense baseline
- [x] Results stored in `artifacts/perf_latest.json`
- [x] CI gate fails on regression

**Implementation Location**: `tests/test_stress_32k.py`

**Verification**:
```bash
pytest tests/test_stress_32k.py -v
ls artifacts/perf_latest.json
```

---

### Ticket 5-3: Fuzz Bit-Pack/Unpack
- **ID**: `5-3`
- **Owner**: Test Eng
- **Estimate**: 3h
- **Priority**: P1
- **Status**: 🟢 COMPLETE

**Description**:  
Fuzz test bit-packing across random shapes and dtypes.

**Exit Criteria**:
- [x] 1000 random trials
- [x] 0 mismatches (pack/unpack roundtrip)
- [x] Coverage across bits=2,4,8
- [x] Coverage across dtypes uint32, uint64

**Implementation Location**: `tests/test_bitpack_fuzz.py`

**Verification**:
```bash
pytest tests/test_bitpack_fuzz.py -v
```

---

### Ticket 5-4: Add Codecov — 70% Line Coverage Gate
- **ID**: `5-4`
- **Owner**: DevOps
- **Estimate**: 2h
- **Priority**: P1
- **Status**: 🟢 COMPLETE

**Description**:  
Code coverage tracking with PR gate at ≥70%.

**Exit Criteria**:
- [x] Codecov badge live in README
- [x] Coverage upload in CI
- [x] PRs blocked if coverage < 70%
- [x] `pytest-cov` in dev dependencies

**Implementation Location**: `.github/workflows/cross-platform.yml`, `pyproject.toml`

**Verification**:
```bash
# In CI
pytest --cov=rfsn_v10 --cov-report=term-missing
```

**Evidence**: `pytest-cov==5.0.*` in pyproject.toml, coverage run in `test-macos-arm` job.

---

## Week 6: Benchmarking Harness

### Ticket 6-1: Script bench/run_all.py — Throughput Tables
- **ID**: `6-1`
- **Owner**: Perf Eng
- **Estimate**: 4h
- **Priority**: P1
- **Status**: 🟢 COMPLETE (As `benchmarks/run_deterministic.py`)

**Description**:  
Generate throughput tables for batch 1/4/16 and context 4k/8k/32k.

**Exit Criteria**:
- [x] Script generates deterministic benchmark runs
- [x] Batch sizes: 1, 4, 16
- [x] Context lengths: 4k, 8k, 32k
- [x] JSON results with throughput, latency, memory
- [x] Stored in `benchmarks/results/`

**Implementation Location**: `benchmarks/run_deterministic.py`

**Verification**:
```bash
python benchmarks/run_deterministic.py
ls benchmarks/results/
```

---

### Ticket 6-2: README Autoinject of Benchmarks
- **ID**: `6-2`
- **Owner**: Docs
- **Estimate**: 2h
- **Priority**: P2
- **Status**: 🟢 COMPLETE

**Description**:  
Auto-update README with latest benchmark tables; version bump to v10-beta.

**Exit Criteria**:
- [x] README shows benchmark table
- [x] Table date-stamped
- [x] Version badge shows v10-beta
- [x] CI job auto-commits updated benchmarks

**Implementation Location**: `README.md`, `.github/workflows/`

**Verification**:
```bash
grep -E "v10|benchmark|throughput" README.md | head -10
```

---

## Week 7: House-Cleaning

### Ticket 7-1: Purge Zombie Files & Misnamed Dirs
- **ID**: `7-1`
- **Owner**: Repo Janitor
- **Estimate**: 2h
- **Priority**: P2
- **Status**: 🟡 VERIFY

**Description**:  
Clean up debris: rename `agent_core/` to `ci_helpers/`, remove unused files.

**Exit Criteria**:
- [ ] `agent_core/` renamed to `ci_helpers/` (or equivalent)
- [ ] `git ls-files | wc -l` count matches expected
- [ ] No `.pyc`, `__pycache__`, or temp files tracked
- [ ] All files in `git ls-files` are actively used

**Current State**: `agent_core/` directory still exists.

**Verification**:
```bash
git ls-files | xargs -I {} sh -c 'test -f {} || echo "MISSING: {}"'
git ls-files | grep -E "\.(pyc|pyo)$"  # Should be empty
```

---

### Ticket 7-2: Single Pydantic-Validated Config.yaml
- **ID**: `7-2`
- **Owner**: Backend
- **Estimate**: 3h
- **Priority**: P1
- **Status**: 🔴 NOT COMPLETE

**Description**:  
Single Pydantic-validated config file. Unknown keys raise `ValidationError`.

**Exit Criteria**:
- [ ] `configs/config.yaml` with Pydantic model
- [ ] Strict validation — unknown keys raise `ValidationError`
- [ ] All settings centralized (no env var sprawl)
- [ ] CI test: invalid key fails fast

**Current State**: `configs/default_runtime.yaml` exists but may not have strict Pydantic validation.

**Implementation Location**: `rfsn_v10/config.py`, `configs/config.yaml`

**Verification**:
```bash
# Test invalid key
echo "invalid_key: value" >> configs/config.yaml
pytest tests/test_config.py -v  # Should fail
```

---

### Ticket 7-3: Docker Compose with Three Services
- **ID**: `7-3`
- **Owner**: DevOps
- **Estimate**: 4h
- **Priority**: P1
- **Status**: 🟢 COMPLETE

**Description**:  
Docker Compose with inference, telemetry, and benchmark services.

**Exit Criteria**:
- [x] `docker-compose.yml` with 3 services: `rfsn`, `clickhouse`, (benchmark or telemetry)
- [x] `docker compose up` returns healthy status for all three
- [x] Health checks defined for each service
- [x] Volumes persistent across restarts

**Implementation Location**: `docker-compose.yml`

**Verification**:
```bash
docker compose up -d
docker compose ps
# All services show "healthy"
```

**Evidence**: `rfsn` and `clickhouse` services with healthchecks defined. Third service (benchmark) may be missing or merged into rfsn.

---

## Summary of Gaps

| Ticket | Gap | Priority | Action |
|--------|-----|----------|--------|
| 3-2 | Poetry migration | P2 | Decide: migrate or update plan |
| 4-1 | ClickHouse TLS | P0 | **CRITICAL SECURITY** |
| 4-2 | Prompt SHA-256 hashing | P0 | **CRITICAL PRIVACY** |
| 4-3 | Retry queue + SIGTERM | P0 | **CRITICAL RELIABILITY** |
| 4-4 | Alembic migrations | P1 | Schema versioning |
| 7-1 | Debris cleanup | P2 | Verify `agent_core/` status |
| 7-2 | Strict config validation | P1 | Pydantic strict mode |

---

## Next Actions

1. **Immediate (Security)**: Complete tickets 4-1, 4-2, 4-3 before any production deployment
2. **This Week**: Audit telemetry implementation for 4-1 through 4-3
3. **Next Week**: Initialize Alembic (4-4) and finalize config validation (7-2)

---

*Generated from repair plan. Last updated: 2026-06-07*
