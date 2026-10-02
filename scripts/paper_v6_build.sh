#!/bin/bash
# Build every paper-v6 table, figure and numerical ledger from the archived results.
#
# Usage: scripts/paper_v6_build.sh OUT
# The inputs are named by environment variables; the defaults are their places on xen1.
# The evidence gate runs first and every renderer that needs it refuses an incomplete report.
set -euo pipefail
OUT=$(realpath -m "$1")
ROOT=/mnt/data/xinyenyana
RESULTS=${RESULTS:-$ROOT/v6-corrected-20260926/results}
HEADLINE=${HEADLINE:-$ROOT/v6run/results}
V5=${V5:-$ROOT/phase-a-20260913/results}
WEIGHTS=${WEIGHTS:-$ROOT/v6run/results/v6-weights.json}
SUPPLEMENTAL=${SUPPLEMENTAL:-$ROOT/v6-corrected-20260926/runs/supplemental-archive-verification.json}
V5_GATE=${V5_GATE:-$ROOT/a1/gate}
# The bat compensation result is archived as gs://xinyenyana/results/af20a2fc...a2d8.json.
BAT=${BAT:?set BAT to a local copy of the archived bat compensation result}
PY=${PY:-python}
cd "$(dirname "$0")/.."
export PYTHONPATH=src:scripts
ENDPOINTS=$($PY -c "from paper_v6_evidence import ENDPOINTS; print(' '.join(ENDPOINTS))")
mkdir -p "$OUT"
EVIDENCE=$OUT/evidence.json

$PY scripts/paper_v6_evidence.py --results "$RESULTS" --headline "$HEADLINE" --output "$EVIDENCE"

$PY scripts/paper_v6_headline.py --results "$HEADLINE" --v5-results "$V5" --out "$OUT/headline"
$PY scripts/paper_v6_headline_agreement.py --results "$RESULTS" --headline "$HEADLINE" \
  --evidence "$EVIDENCE" --out "$OUT/headline/headline-agreement.json"
for ep in $ENDPOINTS; do
  $PY scripts/paper_v6_information.py --results "$RESULTS" --endpoint "$ep" --out "$OUT/information"
done
for ep in $ENDPOINTS; do
  comparison=v6-encoder-comparison.json
  [ "$ep" = great-tit-full ] && comparison=v6-encoder-comparison-great-tit-full.json
  $PY scripts/paper_v6_paired.py --results "$RESULTS" --endpoint "$ep" \
    --comparison "$comparison" --out "$OUT/paired"
done
$PY scripts/paper_v6_overview.py --comparison "$RESULTS/v6-encoder-comparison.json" \
  --out "$OUT/overview"
$PY scripts/paper_v6_main_comparison.py --results "$RESULTS" --components "$OUT" \
  --evidence "$EVIDENCE" --out "$OUT/main-comparison"
$PY scripts/paper_v6_controls.py --results "$RESULTS" --evidence "$EVIDENCE" \
  --out "$OUT/recording-controls"
$PY scripts/paper_v6_main_controls.py --components "$OUT" --evidence "$EVIDENCE" \
  --out "$OUT/main-controls"
$PY scripts/paper_v6_spectral.py --results "$RESULTS" --out "$OUT/spectral"
$PY scripts/paper_v6_additional.py --results "$RESULTS" --out "$OUT/additional"
$PY scripts/paper_v6_open_set.py --results "$RESULTS" --evidence "$EVIDENCE" --out "$OUT/open-set"
$PY scripts/paper_v6_open_roles.py --results "$RESULTS" --evidence "$EVIDENCE" \
  --out "$OUT/open-set"
$PY scripts/paper_v6_candidate_coverage.py --results "$RESULTS" --evidence "$EVIDENCE" \
  --out "$OUT/coverage"
$PY scripts/paper_v6_reproduction.py --results "$RESULTS" --evidence "$EVIDENCE" \
  --supplemental-archive "$SUPPLEMENTAL" --out "$OUT/reproduction"
$PY scripts/paper_v6_model_provenance.py --results "$RESULTS" --weights "$WEIGHTS" \
  --evidence "$EVIDENCE" --out "$OUT/provenance"

# The analyses carried from v5 are drawn by paper_figures.py under their v5 names. The stage
# holds the v5 results, replaced by the v6 headline runs and the carried analyses rerun at the
# v6 counts, so no table in it keeps a v5 count.
STAGE=$OUT/carried-stage
rm -rf "$STAGE"
mkdir -p "$STAGE"
ln -s "$V5"/*.json "$STAGE"/
for f in "$HEADLINE"/v6-headline-*.json; do
  b=$(basename "$f")
  ln -sf "$f" "$STAGE/${b/v6-headline-/v5-identity-}"
done
ln -sf "$RESULTS"/carried/*.json "$STAGE"/
$PY scripts/paper_figures.py --results "$STAGE" --gate "$V5_GATE" --compensation "$BAT" \
  --out "$OUT/carried"
# Quantities added after the fourth review round, and Figure 7, from the ledgers above.
$PY scripts/paper_v7_review.py --build "$OUT" --results "$RESULTS" --v5-results "$V5" \
  --bat "$BAT" --out "$OUT/review"
# The manuscript's tables, in its wording, from the ledgers written above.
$PY scripts/paper_v6_text_tables.py --build "$OUT" --results "$RESULTS" --out "$OUT/paper"
echo "built $OUT"
