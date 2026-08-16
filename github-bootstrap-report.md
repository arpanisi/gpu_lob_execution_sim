# GitHub Bootstrap Report: `gpu_lob_execution_sim`

**Date:** 2026-08-16  
**Repository:** [https://github.com/arpanisi/gpu_lob_execution_sim](https://github.com/arpanisi/gpu_lob_execution_sim)  
**Local Workspace:** `limitbookorder/gpu_lob_execution_sim/`  

---

## 1. Starting State (Inspected & Quoted)

Before executing any changes, the local repository and remote GitHub account were inspected.

### Local Git State
```bash
$ git branch -a && git status && git remote -v
On branch main

No commits yet

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	.DS_Store
	.gitignore
	README.md
	config/
	data/
	docs/
	requirements.txt
	runbook.md
	scripts/
	src/
	tests/

nothing added to commit but untracked files present (use "git add" to track)
```

### Remote GitHub State
```bash
$ gh repo view arpanisi/gpu_lob_execution_sim
GraphQL: Could not resolve to a Repository with the name 'arpanisi/gpu_lob_execution_sim'. (repository)
```

**Conclusion:** The project was an uncommitted local Git repository on `main` with no remotes configured, and no GitHub repository existed yet under `arpanisi/gpu_lob_execution_sim`.

---

## 2. What Already Existed vs. What Was Created

### Pre-Existing (Preserved As-Is)
- Source code in `src/` (`agents/`, `data/`, `env/`, `lob/`, `reporting/`)
- Configuration in `config/models.yaml`
- Test suite in `tests/` (49 passing unit tests covering matching, JAX vectorization, IPPO, order book reconstruction, and evaluation)
- Operational and execution scripts in `scripts/` (`run_tier1_synthetic.py`, `build_reconstructed_range.py`, `run_tier2_ippo.py`, etc.)
- Project documentation in `README.md`, `runbook.md`, and `docs/repo-meta.yaml`
- Architecture diagram in `docs/figure.png`
- `.gitignore` (ignoring `.venv/`, `.env`, Parquet/raw data, cache directories)

### Created and Configured
1. **Initial Git Commit on `main` (`0bb39a5`):** Committed all pre-existing project files to local `main`.
2. **Public GitHub Repository:** Created `arpanisi/gpu_lob_execution_sim` as a public repository and pushed `main`.
3. **`develop` Branch:** Created `develop` off `main` and pushed to remote.
4. **CI Workflow Pipelines (`.github/workflows/`):**
   - `.github/workflows/develop.yml`: Triggers on push and PR to `develop`. Runs `ruff` lint check (fail-fast gate) and fast unit tests (`PYTHONPATH=. pytest tests/ -q`).
   - `.github/workflows/main.yml`: Triggers on push and PR to `main`. Runs full unit tests (`PYTHONPATH=. pytest tests/ -q`), Tier 1 synthetic reconstruction & validation (`PYTHONPATH=. python scripts/run_tier1_synthetic.py ...`), and `ruff` lint check (final gate).
5. **Pull Request #1 (`develop` → `main`):** Opened PR #1 to exercise CI checks on both `develop` and `main` triggers. Both workflows passed successfully (`Lint and Fast Tests` and `Lint and Full Test Suite`). Merged PR #1 into `main`.
6. **Branch Protection on `main`:** Enforced strict status checks (`Lint and Full Test Suite`), PR requirement, no direct pushes, no force pushes, and no branch deletions.
7. **Branch Protection on `develop`:** Enforced status check (`Lint and Fast Tests`), no force pushes, and no branch deletions.
8. **Repository Metadata:** Applied About description and topics from `docs/repo-meta.yaml`.

---

## 3. Real Command Output for GitHub-Side Operations

### Step 2: Initial Commit & GitHub Repository Creation
```bash
$ git commit -m "feat: initial codebase for gpu lob execution sim"
[main (root-commit) 0bb39a5] feat: initial codebase for gpu lob execution sim
 59 files changed, 4107 insertions(+)
 create mode 100644 .gitignore
 create mode 100644 README.md
 create mode 100644 config/models.yaml
 create mode 100644 data/.gitkeep
 create mode 100644 docs/.gitkeep
 create mode 100644 docs/figure.png
 create mode 100644 docs/repo-meta.yaml
 create mode 100644 outputs/.gitkeep
 create mode 100644 requirements.txt
 create mode 100644 runbook.md
 create mode 100644 scripts/build_reconstructed_range.py
 create mode 100644 scripts/measure_step3_speedup.py
 create mode 100644 scripts/prepare_tier2_ippo.py
 create mode 100644 scripts/prepare_tier3_full_run.py
 create mode 100644 scripts/probe_bybit_sources.py
 create mode 100644 scripts/reconstruct_events.py
 create mode 100644 scripts/report_time_of_day.py
 create mode 100644 scripts/run_tier1_replay.py
 create mode 100644 scripts/run_tier1_synthetic.py
 create mode 100644 scripts/run_tier2_ippo.py
 create mode 100644 src/agents/ippo_config.py
 create mode 100644 src/agents/jax_ippo.py
 create mode 100644 src/data/date_ranges.py
 create mode 100644 src/data/event_io.py
 create mode 100644 src/data/informed_flow.py
 create mode 100644 src/data/orderbook.py
 create mode 100644 src/data/reconstruct.py
 create mode 100644 src/data/sources.py
 create mode 100644 src/data/trades.py
 create mode 100644 src/data/validation.py
 create mode 100644 src/env/episode.py
 create mode 100644 src/env/execution.py
 create mode 100644 src/env/jax_batched_env.py
 create mode 100644 src/env/observation.py
 create mode 100644 src/env/windows.py
 create mode 100644 src/lob/constants.py
 create mode 100644 src/lob/events.py
 create mode 100644 src/lob/jax_matching.py
 create mode 100644 src/lob/matching.py
 create mode 100644 src/reporting/evaluation_io.py
 create mode 100644 src/reporting/time_of_day.py
 create mode 100644 tests/conftest.py
 create mode 100644 tests/test_build_reconstructed_range.py
 create mode 100644 tests/test_date_ranges.py
 create mode 100644 tests/test_environment.py
 create mode 100644 tests/test_evaluation_reporting.py
 create mode 100644 tests/test_event_io_validation.py
 create mode 100644 tests/test_informed_flow.py
 create mode 100644 tests/test_jax_batched_env.py
 create mode 100644 tests/test_jax_ippo.py
 create mode 100644 tests/test_jax_matching.py
 create mode 100644 tests/test_matching.py
 create mode 100644 tests/test_observation.py
 create mode 100644 tests/test_orderbook_parser.py
 create mode 100644 tests/test_reconstruct.py
 create mode 100644 tests/test_reporting_and_config.py
 create mode 100644 tests/test_sources.py
 create mode 100644 tests/test_tier3_preflight.py
 create mode 100644 tests/test_windows.py

$ gh repo create gpu_lob_execution_sim --public --source=. --remote=origin --push
https://github.com/arpanisi/gpu_lob_execution_sim
To https://github.com/arpanisi/gpu_lob_execution_sim.git
 * [new branch]      HEAD -> main
branch 'main' set up to track 'origin/main'.
```

### Step 3: Create and Push `develop` Branch
```bash
$ git checkout -b develop && git push -u origin develop
Switched to a new branch 'develop'
To https://github.com/arpanisi/gpu_lob_execution_sim.git
 * [new branch]      develop -> develop
branch 'develop' set up to track 'origin/develop'.
```

### Step 4: Add CI Workflows and Push via `develop`
```bash
$ git add .github/ && git commit -m "ci: add develop and main workflow pipelines" && git push origin develop
[develop 127b79d] ci: add develop and main workflow pipelines
 2 files changed, 74 insertions(+)
 create mode 100644 .github/workflows/develop.yml
 create mode 100644 .github/workflows/main.yml
To https://github.com/arpanisi/gpu_lob_execution_sim.git
   0bb39a5..127b79d  develop -> develop
```

### Step 5: Open PR #1 (`develop` → `main`) and Verify CI
```bash
$ gh pr create --base main --head develop --title "CI: Initialize develop and main workflows" --body "Sets up develop and main CI workflows per quantprojects workflow plan."
https://github.com/arpanisi/gpu_lob_execution_sim/pull/1

$ gh pr checks 1
Lint and Fast Tests         pass    48s    https://github.com/arpanisi/gpu_lob_execution_sim/actions/runs/31931542054/job/95127028883
Lint and Full Test Suite    pass    45s    https://github.com/arpanisi/gpu_lob_execution_sim/actions/runs/31931542866/job/95127030937
```

### Merge PR #1 into `main`
```bash
$ gh pr merge 1 --merge --subject "ci: initialize develop and main workflows"
```

### Step 6: Configure Branch Protection
```bash
$ gh api --method PUT repos/arpanisi/gpu_lob_execution_sim/branches/main/protection \
  --input - << 'EOF'
{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "Lint and Full Test Suite"
    ]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": false,
    "require_code_owner_reviews": false,
    "required_approving_review_count": 0
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
EOF

$ gh api --method PUT repos/arpanisi/gpu_lob_execution_sim/branches/develop/protection \
  --input - << 'EOF'
{
  "required_status_checks": {
    "strict": false,
    "contexts": [
      "Lint and Fast Tests"
    ]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": null,
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
EOF
```

### Step 7: Apply About Description and Topics from `docs/repo-meta.yaml`
```bash
$ gh repo edit arpanisi/gpu_lob_execution_sim \
  --description "GPU-batched reinforcement learning for limit-order-book optimal execution with IPPO, trained against Bybit BTCUSDT L2 order-book data." \
  --add-topic "limit-order-book,optimal-execution,ippo,price-time-priority,implementation-shortfall,informed-flow"
```

---

## 4. Final Branch Protection State (Fetched Back from GitHub API)

### `main` Branch Protection (`gh api repos/arpanisi/gpu_lob_execution_sim/branches/main/protection`)
```json
{
  "url": "https://api.github.com/repos/arpanisi/gpu_lob_execution_sim/branches/main/protection",
  "required_status_checks": {
    "url": "https://api.github.com/repos/arpanisi/gpu_lob_execution_sim/branches/main/protection/required_status_checks",
    "strict": true,
    "contexts": [
      "Lint and Full Test Suite"
    ],
    "contexts_url": "https://api.github.com/repos/arpanisi/gpu_lob_execution_sim/branches/main/protection/required_status_checks/contexts",
    "checks": [
      {
        "context": "Lint and Full Test Suite",
        "app_id": 15368
      }
    ]
  },
  "required_pull_request_reviews": {
    "url": "https://api.github.com/repos/arpanisi/gpu_lob_execution_sim/branches/main/protection/required_pull_request_reviews",
    "dismiss_stale_reviews": false,
    "require_code_owner_reviews": false,
    "require_last_push_approval": false,
    "required_approving_review_count": 0
  },
  "required_signatures": {
    "url": "https://api.github.com/repos/arpanisi/gpu_lob_execution_sim/branches/main/protection/required_signatures",
    "enabled": false
  },
  "enforce_admins": {
    "url": "https://api.github.com/repos/arpanisi/gpu_lob_execution_sim/branches/main/protection/enforce_admins",
    "enabled": false
  },
  "required_linear_history": {
    "enabled": false
  },
  "allow_force_pushes": {
    "enabled": false
  },
  "allow_deletions": {
    "enabled": false
  },
  "block_creations": {
    "enabled": false
  },
  "required_conversation_resolution": {
    "enabled": false
  },
  "lock_branch": {
    "enabled": false
  },
  "allow_fork_syncing": {
    "enabled": false
  }
}
```

### `develop` Branch Protection (`gh api repos/arpanisi/gpu_lob_execution_sim/branches/develop/protection`)
```json
{
  "url": "https://api.github.com/repos/arpanisi/gpu_lob_execution_sim/branches/develop/protection",
  "required_status_checks": {
    "url": "https://api.github.com/repos/arpanisi/gpu_lob_execution_sim/branches/develop/protection/required_status_checks",
    "strict": false,
    "contexts": [
      "Lint and Fast Tests"
    ],
    "contexts_url": "https://api.github.com/repos/arpanisi/gpu_lob_execution_sim/branches/develop/protection/required_status_checks/contexts",
    "checks": [
      {
        "context": "Lint and Fast Tests",
        "app_id": 15368
      }
    ]
  },
  "required_signatures": {
    "url": "https://api.github.com/repos/arpanisi/gpu_lob_execution_sim/branches/develop/protection/required_signatures",
    "enabled": false
  },
  "enforce_admins": {
    "url": "https://api.github.com/repos/arpanisi/gpu_lob_execution_sim/branches/develop/protection/enforce_admins",
    "enabled": false
  },
  "required_linear_history": {
    "enabled": false
  },
  "allow_force_pushes": {
    "enabled": false
  },
  "allow_deletions": {
    "enabled": false
  },
  "block_creations": {
    "enabled": false
  },
  "required_conversation_resolution": {
    "enabled": false
  },
  "lock_branch": {
    "enabled": false
  },
  "allow_fork_syncing": {
    "enabled": false
  }
}
```

---

## 5. Repository Metadata Verification (Fetched Back from GitHub API)

### Queried via `gh repo view arpanisi/gpu_lob_execution_sim --json description,repositoryTopics,name,isPrivate,defaultBranchRef`
```json
{
  "defaultBranchRef": {
    "name": "main"
  },
  "description": "GPU-batched reinforcement learning for limit-order-book optimal execution with IPPO, trained against Bybit BTCUSDT L2 order-book data.",
  "isPrivate": false,
  "name": "gpu_lob_execution_sim",
  "repositoryTopics": [
    {
      "name": "implementation-shortfall"
    },
    {
      "name": "informed-flow"
    },
    {
      "name": "ippo"
    },
    {
      "name": "limit-order-book"
    },
    {
      "name": "optimal-execution"
    },
    {
      "name": "price-time-priority"
    }
  ]
}
```

### Comparison against `docs/repo-meta.yaml`
- **Description:** Exact match  
  `"GPU-batched reinforcement learning for limit-order-book optimal execution with IPPO, trained against Bybit BTCUSDT L2 order-book data."`
- **Topics:** Exact match  
  `['implementation-shortfall', 'informed-flow', 'ippo', 'limit-order-book', 'optimal-execution', 'price-time-priority']` vs `['limit-order-book', 'optimal-execution', 'ippo', 'price-time-priority', 'implementation-shortfall', 'informed-flow']` (GitHub sorts topics).
- **Visibility:** Public (`isPrivate: false`).

---

## 6. GitHub Actions Workflow Execution Summary

All workflow runs completed successfully:

| Run ID | Workflow | Event | Branch / Ref | Duration | Status | Check Context |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `31931542054` | Develop CI | `push` | `develop` | 51s | **Success** | `Lint and Fast Tests` |
| `31931542866` | Main CI | `pull_request` | PR #1 (`develop` → `main`) | 48s | **Success** | `Lint and Full Test Suite` |
| `31931592802` | Main CI | `push` | `main` (merge commit) | 48s | **Success** | `Lint and Full Test Suite` |

Both local and remote branches (`main` and `develop`) are synchronized, carry both `.github/workflows/` files, and follow the complete quantprojects CI & branch protection specification.
