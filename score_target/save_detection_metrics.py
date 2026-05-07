import pandas as pd
import re
from pathlib import Path
import time
from tagging import tag_target, extract_noun
import numpy as np
import pickle
from nltk.stem import PorterStemmer
import os
from matplotlib import pyplot as plt
from matplotlib import rcParams

stemmer = PorterStemmer()

home = str(Path.home())

PATH = "detection_results/"

files = [
    "deepseek-R1.csv",
    "gpt-4o.csv",
]

def plot_sims(dataset_sims: np.array, specificity_sims: np.array, model_name: str, is_not: bool) -> None:
    rcParams['figure.figsize'] = 8, 6
    plt.hist(specificity_sims, bins=25, color='g', edgecolor='w', alpha=0.5, label="Similarity With Specific Target")
    plt.hist(dataset_sims, bins=25, color='orange', edgecolor='w', alpha=0.5, label="Similarity With Dataset Target")
    plt.grid(axis='y', alpha=0.3)
    plt.legend(fontsize=18)
    model_name = model_name.replace("deepseek", "DeepSeek").replace("gpt", "GPT")
    if is_not:
        plt.title(f"{model_name} Specificity Distributions Among Non-matching Strings", fontsize=21)
        plt.savefig(f"{model_name}_specificity_comparision_nonmatch.png", format="png")
    else:
        plt.title(f"{model_name} Specificity Distributions", fontsize=21)
        plt.savefig(f"{model_name}_specificity_comparision.png", format="png")
    plt.xlabel("Cosine Similarity")
    plt.close()

if __name__ == "__main__":
    rows = []
    for file in files:
        curr = pd.read_csv(PATH + file)
        dataset_match_bin = len(curr[curr["dataset_match"] == True])
        specific_match_bin = len(curr[curr["specific_match"] == True])
        other_match_bin = len(curr) - dataset_match_bin - specific_match_bin
        dataset_more_sim = len(curr[curr["dataset_sim"] > curr["specific_sim"]])
        specific_more_sim = len(curr[curr["specific_sim"] > curr["dataset_sim"]])
        not_string_match = curr[(curr["specific_match"] == False) & (curr["dataset_match"] == False)]
        avg_dataset_sim = curr["dataset_sim"].mean()
        avg_not_match_dataset_sim = not_string_match["dataset_sim"].mean()
        avg_specific_sim = curr["specific_sim"].mean()
        avg_not_match_specific_sim = not_string_match["specific_sim"].mean()
        sd_dataset_sim = curr["dataset_sim"].std()
        sd_not_match_dataset_sim = not_string_match["dataset_sim"].std()
        sd_specific_sim = curr["specific_sim"].std()
        sd_not_match_specific_sim = not_string_match["specific_sim"].std()
        model_name = file[:file.index(".csv")]
        plot_sims(curr["dataset_sim"].to_numpy(), curr["specific_sim"].to_numpy(), model_name, False)
        plot_sims(not_string_match["dataset_sim"].to_numpy(), not_string_match["specific_sim"].to_numpy(), model_name, True)
        print(f'''For {model_name}:\n
Dataset Target Matches: {dataset_match_bin}\n
Specific Target Matches: {specific_match_bin}\n
Neither Target Matches: {other_match_bin}\n
More Similar To Dataset Target: {dataset_more_sim}\n
More Similar To Specific Target: {specific_more_sim}\n
Avg Dataset Cosine Similarity: {avg_dataset_sim}\n
Avg Specific Cosine Similarity: {avg_specific_sim}\n
SD Dataset Cosine Similarity: {sd_dataset_sim}\n
SD Specific Cosine Similarity: {sd_specific_sim}\n
Avg Dataset Cosine Similarity (Non-matching Strings): {avg_not_match_dataset_sim}\n
Avg Specific Cosine Similarity (Non-matching Strings): {avg_not_match_specific_sim}\n
SD Dataset Cosine Similarity (Non-matching Strings): {sd_not_match_dataset_sim}\n
SD Specific Cosine Similarity (Non-matching Strings): {sd_not_match_specific_sim}\n''')

        rows.append({
            "model": model_name,
            "dataset_target_matches": dataset_match_bin / len(curr),
            "specific_target_matches": specific_match_bin / len(curr),
            "neither_target_matches": other_match_bin / len(curr),
            "more_similar_to_dataset": dataset_more_sim / len(curr),
            "more_similar_to_specific": specific_more_sim / len(curr),
            "avg_dataset_sim": avg_dataset_sim,
            "avg_specific_sim": avg_specific_sim,
            "sd_dataset_sim": sd_dataset_sim,
            "sd_specific_sim": sd_specific_sim,
            "avg_not_match_dataset_sim": avg_not_match_dataset_sim,
            "avg_not_match_specific_sim": avg_not_match_specific_sim,
            "sd_not_match_dataset_sim": sd_not_match_dataset_sim,
            "sd_not_match_specific_sim": sd_not_match_specific_sim
        })

    summary = pd.DataFrame(rows)
    os.makedirs(PATH + "summary", exist_ok=True)
    summary.to_csv(PATH + "summary/summary.csv", index=False)

