#!/bin/bash
set -e

echo "Starting full experiment pipeline..."

DATASETS=(
    "open" "open_cot" "mcq_2" "mcq_4" "mcq_none_or_all"
    "open_source" "open_source_stage_2"
    "antonym_original_target" "antonym_our_target" "baseline_our_target"
    "baseline_original_target" "pseudoword_our_target" "pseudoword_original_target"
    "v_source" "n_source"
)


echo "Phase 1: Getting all responses"

for dataset in "${DATASETS[@]}"; do
    echo "-> Getting responses for: $dataset"
    python main.py get_responses --dataset "$dataset"
done


echo "Phase 2: Summarizing all responses"

for dataset in "${DATASETS[@]}"; do
    echo "-> Summarizing responses for: $dataset"
    python main.py summarize_responses --dataset "$dataset"
done


echo "Phase 3: Running Global Metrics"

echo "-> Getting target specificity..."
python main.py get_target_specificity

echo "-> Getting open source metrics..."
python main.py get_open_source_metrics

echo "-> Getting detection metrics..."
python main.py get_detection_metrics

echo "Pipeline completed successfully."
