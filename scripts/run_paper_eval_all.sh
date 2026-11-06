#!/usr/bin/env bash
# Run the EvoMDT evaluation. Every step resumes where it stopped.
#   PER_CANCER   Layer 2 cases per cancer (default 50, so 200 in total). Empty means every case.
#   RUN_EXTRAS=1 also runs HCC-TACE-Seg and the Layer 1 benchmarks, which add many hours.
# Set PY to the project's Python, for example: PY=.venv/Scripts/python.exe bash scripts/run_paper_eval_all.sh
set -eu
cd "$(dirname "$0")/.."
PY="${PY:-.venv/Scripts/python.exe}"
PER_CANCER="${PER_CANCER-50}"
RUN_EXTRAS="${RUN_EXTRAS:-0}"
LOG=artifacts/paper_eval
REAL="real_breast_cancer real_hepatocellular_carcinoma real_lung_adenocarcinoma real_lymphoma"
DEV="dev_medqa dev_real_breast_cancer dev_real_hepatocellular_carcinoma dev_real_lung_adenocarcinoma dev_real_lymphoma"
LAYER1="medqa medqa_cancer medmcqa_cancer medquad rond"
BACKBONE="nvidia/nemotron-3-ultra-550b-a55b"
RUN_CAP="${PER_CANCER:+--per-file $PER_CANCER}"
SCORE_CAP="${PER_CANCER:+--per-cancer $PER_CANCER}"
BASELINES="single:$BACKBONE"

step() { echo "=== $(date '+%H:%M:%S') $*"; }

# The independent audit of extractions runs on its own (it needs a slow model and must not block this chain):
#   python -P scripts/prepare_paper_data.py audit --concurrency 3

if [ ! -f "$LOG/evolution_state.json" ]; then
  step "evolve on the development cases, then freeze the state"
  "$PY" scripts/run_paper_eval.py evolve --files $DEV --concurrency 10 --rpm 40 --timeout 600 --batch 8 > "$LOG/evolve.log" 2>&1
  tail -12 "$LOG/evolve.log"
fi

# The judges and the single-agent arm use little of the backbone's capacity, so they run beside EvoMDT.
step "Layer 2: single-agent backbone and first judge pass (in the background)"
"$PY" scripts/run_paper_eval.py run --system "single:$BACKBONE" --files $REAL $RUN_CAP --concurrency 4 --rpm 15 > "$LOG/single_real.log" 2>&1 &
SINGLE=$!
"$PY" scripts/score_paper_eval.py judge $SCORE_CAP --concurrency 6 > "$LOG/judge.log" 2>&1 &
JUDGES=$!

step "Layer 2: EvoMDT"
"$PY" scripts/run_paper_eval.py run --system evomdt --files $REAL $RUN_CAP --concurrency 10 --rpm 40 --timeout 600 > "$LOG/evomdt_real.log" 2>&1
tail -1 "$LOG/evomdt_real.log"
wait $SINGLE
tail -1 "$LOG/single_real.log"

if [ "$RUN_EXTRAS" = "1" ]; then
  BASELINES="$BASELINES single:mistralai/mistral-large-4-1t-a52b"
  step "HCC-TACE-Seg: EvoMDT"
  "$PY" scripts/run_paper_eval.py run --system evomdt_hcc_tace --files hcc_tace --concurrency 10 --rpm 40 --timeout 600 > "$LOG/evomdt_hcc.log" 2>&1
  tail -1 "$LOG/evomdt_hcc.log"

  step "Layer 1: EvoMDT"
  "$PY" scripts/run_paper_eval.py run --system evomdt --files $LAYER1 --concurrency 10 --rpm 40 --timeout 600 > "$LOG/evomdt_layer1.log" 2>&1
  tail -1 "$LOG/evomdt_layer1.log"

  step "Layer 1: single-agent backbone"
  "$PY" scripts/run_paper_eval.py run --system "single:$BACKBONE" --files $LAYER1 --concurrency 12 --rpm 40 > "$LOG/single_layer1.log" 2>&1
  tail -1 "$LOG/single_layer1.log"

  step "Layer 1: Mistral comparison on the cancer-focused sets"
  "$PY" scripts/run_paper_eval.py run --system "single:mistralai/mistral-large-4-1t-a52b" --files medqa_cancer medmcqa_cancer rond --concurrency 6 --rpm 40 > "$LOG/single_mistral.log" 2>&1
  tail -1 "$LOG/single_mistral.log"
fi

wait $JUDGES
step "judges, remaining tasks"
"$PY" scripts/score_paper_eval.py judge $SCORE_CAP --concurrency 6 > "$LOG/judge2.log" 2>&1
tail -2 "$LOG/judge2.log"

step "report"
"$PY" scripts/score_paper_eval.py report $SCORE_CAP --baselines $BASELINES > "$LOG/report.log" 2>&1
tail -3 "$LOG/report.log"
step "finished"
