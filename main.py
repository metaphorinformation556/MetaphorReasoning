'''
File locations:
questions/ask_llm_questions_vllm.py
questions/summary_utils.py
score_target/score_and_compare_open_questions.py
source_questions/second_stage_scoring.py

args to execute each file from this file:
--get_responses -> questions/ask_llm_questions.py
--summarize_responses -> questions/summary_utils.py
--get_target_specificity -> score_target/score_and_compare_open_questions.py
--get_open_source_metrics -> source_questions/second_stage_scoring.py

args per file:
questions/ask_llm_questions.py -> ["open", "mcq_2", "mcq_4", "mcq_seojin", "open_source", "open_source_stage_2", "baseline_mapping", "antonym_mapping", "pseudoword_mapping", "v_source", "n_source"]
questions/summary_utils.py -> ["open", "mcq_2", "mcq_4", "mcq_none_or_all", "open_source", "open_source_stage_2", "baseline_mapping", "antonym_mapping", "pseudoword_mapping", "v_source", "n_source"]
score_target/score_and_compare_open_questions.py -> N/A
source_questions/second_stage_scoring.py -> N/A
'''

import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent

SCRIPT_PATHS = {
    "get_responses": BASE_DIR / "questions" / "ask_llm_questions_vllm.py",
    "summarize_responses": BASE_DIR / "questions" / "summary_utils.py",
    "get_target_specificity": BASE_DIR / "score_target" / "score_and_compare_open_questions.py",
    "get_open_source_metrics": BASE_DIR / "source_questions" / "second_stage_scoring.py",
}

VALID_DATASETS = [
    "open", "open_cot", "mcq_2", "mcq_4", "mcq_seojin",
    "open_source", "open_source_stage_2",
    "antonym_original_target", "antonym_our_target", "baseline_our_target"
    ,"baseline_original_target", "pseudoword_our_target"
    ,"pseudoword_original_target" , "v_source", "n_source"
]


def run_script(script_path, child_args):
    cmd = [sys.executable, str(script_path)] + child_args

    print(f"Running: {' '.join(cmd)}")
    print(f"CWD: {script_path.parent}")

    return subprocess.run(cmd, cwd= script_path.parent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest= "command", required= True)

    #---- get_responses ----
    parser_get = subparsers.add_parser("get_responses")
    parser_get.add_argument(
        "--dataset",
        choices= VALID_DATASETS,
        required= True
    )

    #---- summarize_responses ----
    parser_sum = subparsers.add_parser("summarize_responses")
    parser_sum.add_argument(
        "--dataset",
        choices= VALID_DATASETS,
        required= True
    )

    #---- get_target_specificity ----
    subparsers.add_parser("get_target_specificity")

    #---- get_open_source_metrics ----
    subparsers.add_parser("get_open_source_metrics")

    args = parser.parse_args()

    command = args.command
    script_path = SCRIPT_PATHS[command]

    #build child args
    child_args = []

    if command in ["get_responses", "summarize_responses"]:
        #pass dataset as positional argument
        child_args.extend(["--type", args.dataset])

    result = run_script(script_path, child_args)

    if result.returncode != 0:
        print(f"{command} failed with exit code {result.returncode}")