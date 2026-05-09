#!/usr/bin/env bash

set -e

ENV_NAME="uv-research"

NOTEBOOKS=(
# "notebooks/eda/EDA.ipynb"
# "notebooks/regression/regression.ipynb"
# "notebooks/regression/regression_modern.ipynb"
#  "notebooks/regression/regression_hybrid.ipynb"
# "notebooks/recommendation/recommendation.ipynb"
)

format_time () {
  local T=$1
  printf "%02d:%02d:%02d" $((T/3600)) $(( (T%3600)/60 )) $((T%60))
}

TOTAL_START_TS=$(date +"%Y-%m-%d %H:%M:%S")
TOTAL_START=$SECONDS

echo "========================================"
echo "Pipeline started at: $TOTAL_START_TS"
echo "========================================"

for nb in "${NOTEBOOKS[@]}"; do
  echo "----------------------------------------"

  START_TS=$(date +"%Y-%m-%d %H:%M:%S")
  START=$SECONDS

  echo "Executing: $nb"
  echo "Start time: $START_TS"

  conda run -n $ENV_NAME jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace "$nb"

  END_TS=$(date +"%Y-%m-%d %H:%M:%S")
  DURATION=$((SECONDS - START))

  echo "End time:   $END_TS"
  echo "Duration:   $(format_time $DURATION)"
done

TOTAL_END_TS=$(date +"%Y-%m-%d %H:%M:%S")
TOTAL_DURATION=$((SECONDS - TOTAL_START))

echo "========================================"
echo "COLLECTING RESULTS TO EXCEL"
echo "========================================"

conda run -n $ENV_NAME python scripts/collect_results.py

if [$? -eq 0]; then
  echo "Successful"
else
  echo "FAILURE"
fi

echo "========================================"
echo "Pipeline finished at: $TOTAL_END_TS"
echo "Total duration: $(format_time $TOTAL_DURATION)"
echo "========================================"