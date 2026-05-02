#!/usr/bin/env bash

set -e
set -o pipefail

PYTHON=python3

GET_RESPONSES_DATASETS=(
  "open"
  "mcq_2"
  "mcq_4"
  "open_source"
  "open_source_stage_2"
  "baseline_mapping"
  "antonym_mapping"
  "pseudoword_mapping"
  "v_source"
  "n_source"
)

SUMMARIZE_DATASETS=(
  "open"
  "mcq_2"
  "mcq_4"
  "mcq_none_or_all"
  "open_source"
  "open_source_stage_2"
  "baseline_mapping"
  "antonym_mapping"
  "pseudoword_mapping"
  "v_source"
  "n_source"
)

echo "Running: get_responses"

for ds in "${GET_RESPONSES_DATASETS[@]}"; do
  echo "Dataset: $ds"
  $PYTHON questions/ask_llm_questions_vllm.py --type "$ds"
done

echo "Running: summarize_responses"

for ds in "${SUMMARIZE_DATASETS[@]}"; do
  echo "Dataset: $ds"
  $PYTHON questions/summary_utils.py --type "$ds"
done

echo "Running: get_target_specificity"

$PYTHON score_target/score_and_compare_open_questions.py

echo "Running: get_open_source_metrics"

$PYTHON source_questions/second_stage_scoring.py

echo "Running: get_detection_metrics"

$PYTHON score_target/get_detection_metrics.py

echo "Done"