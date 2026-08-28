#!/usr/bin/env bash
#
# Reproduce every computed result in the paper, in order.
#
#   bash scripts/run_all.sh
#
# Runs on CPU in roughly two minutes. Step 01 verifies the inputs and the whole
# run aborts if it fails, so no later step can analyse a different data snapshot.
#
# Outputs land in results/tables/reproduction_*.csv and results/figures/.
# Each script prints its recomputed value beside the published one and states the
# divergence; see the "Known differences on re-running" section of README.md.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "=============================================================="
echo " 01  verify inputs (aborts the run on any mismatch)"
echo "=============================================================="
python scripts/01_verify_inputs.py

echo
echo "=============================================================="
echo " 02  provenance noise floor  (the paper's central control)"
echo "=============================================================="
python scripts/02_provenance_floor.py

echo
echo "=============================================================="
echo " 03  naive -> honest AUC collapse for all three comparisons"
echo "=============================================================="
python scripts/03_pairwise_collapse.py

echo
echo "=============================================================="
echo " 04  verify the frozen bundle and re-run the worked example"
echo "=============================================================="
python scripts/04_verify_bundle.py

echo
echo "=============================================================="
echo " 05-17  regenerate all 13 figures from the shipped tables"
echo "=============================================================="
# Each script reads only results/tables/ and data/processed/ and writes one PNG.
# results/figures/README.md maps every figure to its script and inputs.
for s in scripts/05_figure6.py \
         scripts/06_figure1_dataset.py \
         scripts/07_figure2_provenance_floor.py \
         scripts/08_figure3_collapse.py \
         scripts/09_figure4_chemspace.py \
         scripts/10_figure5_scaffolds_alerts.py \
         scripts/11_figureS1_curation.py \
         scripts/12_figureS2_benchmark.py \
         scripts/13_figureS3_yscrambling.py \
         scripts/14_figureS4_shap.py \
         scripts/15_figureS5_leakage.py \
         scripts/16_figureS6_alerts_by_document.py \
         scripts/17_figureS7_conformal.py ; do
    printf '  %-42s' "$(basename "$s")"
    python "$s" >/dev/null && echo "OK" || { echo "FAILED"; exit 1; }
done

echo
echo "=============================================================="
echo " tests: the app must agree with the published worked example"
echo "=============================================================="
python -m pytest app/test_app.py -q --no-header

echo
echo "All steps completed. Compare the printed values against README.md"
echo "'Known differences on re-running' before reporting a discrepancy."
