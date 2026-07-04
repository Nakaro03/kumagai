#!/usr/bin/env bash
# Full X5 evaluation pipeline (sequential, single-GPU):
#   Phase 1 : Full sweep            (15 runs ≈ 2.1 hr)
#   Phase 2 : seed=42 patch         ( 3 runs ≈ 26 min)   ← lost to a smoke-test skip
#   Phase 3 : Trajectory baselines  (30 runs ≈ 4.3 hr)   PRESCIENT + MIOFlow
#   Phase 4 : Ablation A1..A6       (90 runs ≈ 12.8 hr)
#   Phase 5 : Aggregate to RESULTS_X5/SUMMARY/*.md
#
# Each phase is idempotent — completed evaluation.json files are skipped if
# their saved epochs >= requested.
#
# Run:  pnode_patent_runner/scripts/run_x5_full_pipeline.sh
# Skip phases:  SKIP_PHASE1=1 SKIP_PHASE3=1 ... before invoking

set -u
cd /home/nakamuraroi/kumagai
mkdir -p RESULTS_X5/SUMMARY logs_x5_pipeline

PIPELINE_LOG="logs_x5_pipeline/master.log"
echo "=== X5 pipeline start: $(date) ===" | tee "$PIPELINE_LOG"

run_phase () {
  local name="$1"; shift
  local cmd="$*"
  echo "" | tee -a "$PIPELINE_LOG"
  echo "[$(date +%H:%M:%S)] >>> $name" | tee -a "$PIPELINE_LOG"
  echo "    $cmd" | tee -a "$PIPELINE_LOG"
  bash -c "$cmd" >> "$PIPELINE_LOG" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] <<< $name  (rc=$rc)" | tee -a "$PIPELINE_LOG"
}

# ── Phase 1: full X5 sweep (15 runs) ───────────────────────────────────────
if [[ "${SKIP_PHASE1:-0}" != "1" ]]; then
  run_phase "Phase 1: full X5 sweep" \
    "bash pnode_patent_runner/scripts/run_x5_sweep.sh"
fi

# ── Phase 2: patch seed=42 patent_energy (skip-bug salvage) ────────────────
if [[ "${SKIP_PHASE2:-0}" != "1" ]]; then
  run_phase "Phase 2: seed=42 patent_energy patch" \
    "FORCE=1 PNODE_SEEDS=42 DOMAINS=patent_energy_top50 bash pnode_patent_runner/scripts/run_x5_sweep.sh"
fi

# ── Phase 3: trajectory baselines (30 runs) ────────────────────────────────
if [[ "${SKIP_PHASE3:-0}" != "1" ]]; then
  run_phase "Phase 3: PRESCIENT + MIOFlow baselines" \
    "bash pnode_patent_runner/scripts/run_trajectory_baselines.sh"
fi

# ── Phase 4: ablation A1..A6 (90 runs; A0 is Phase 1) ──────────────────────
if [[ "${SKIP_PHASE4:-0}" != "1" ]]; then
  run_phase "Phase 4: ablation A1..A6" \
    "ABL_ORDER='A1 A2 A3 A4 A5 A6' bash pnode_patent_runner/scripts/run_x5_ablations.sh"
fi

# ── Phase 5: aggregate (descriptive sweep) ─────────────────────────────────
if [[ "${SKIP_PHASE5:-0}" != "1" ]]; then
  run_phase "Phase 5a: full-sweep table" \
    "python pnode_patent_runner/aggregate_x5_full.py --out RESULTS_X5/SUMMARY/full_sweep.md"
  run_phase "Phase 5b: ablation table" \
    "python pnode_patent_runner/aggregate_x5_ablations.py --out RESULTS_X5/SUMMARY/ablations.md"
  run_phase "Phase 5c: unified Table 1" \
    "python pnode_patent_runner/aggregate_x5_unified.py --out RESULTS_X5/SUMMARY/TABLE1.md"
fi

# ── Phase 6: TRUE leave-one-out evaluation (predictive verdict) ────────────
if [[ "${SKIP_PHASE6:-0}" != "1" ]]; then
  run_phase "Phase 6: X5 true held-out LOO sweep" \
    "bash pnode_patent_runner/scripts/run_x5_loo_sweep.sh"
  run_phase "Phase 6b: LOO aggregate" \
    "python pnode_patent_runner/aggregate_x5_loo.py --out RESULTS_X5/SUMMARY/LOO_VERDICT.md"
fi

echo "" | tee -a "$PIPELINE_LOG"
echo "=== X5 pipeline DONE: $(date) ===" | tee -a "$PIPELINE_LOG"
