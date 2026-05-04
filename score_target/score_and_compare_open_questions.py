import pandas as pd
import os
from scoring_functions import get_first_score, get_third_score
from sklearn.preprocessing import MinMaxScaler
from graph_specificity_scores import graph_data
import re
from tagging import tag_target, extract_noun
from sklearn.preprocessing import MinMaxScaler
from get_statistics import get_mannwhitney_p_value, get_t_p_value, get_spearman_corr, get_pearson_corr, get_ks_p_value
import pickle
import numpy as np
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
from pathlib import Path
home = str(Path.home())

#get dict of conceptnet embeddings
def get_conceptnet_embeddings() -> dict:
    with open(home + "/MetaphorReasoning/conceptnet_embeddings/embeddings.pkl", "rb") as f:
        conceptnet_embeddings = pickle.load(f)
        print("Finished loading ConceptNet embeddings...\n")
        return conceptnet_embeddings

#calculate cosine similarity between two embeddings
def cos_sim(embedding_1: np.array, embedding_2: np.array) -> float:
    return np.dot(embedding_1, embedding_2) / (np.linalg.norm(embedding_1) * np.linalg.norm(embedding_2))

def get_features(word: str, embeddings: dict):
    """
    Return a numpy embedding for a word or compound word.

    1 Exact match of the full word
    2 If compound (joined by '_' or '-'), average embeddings of each part using exact or stem-based lookup
    """
    word = word.lower().strip().replace(".", "")
    embedding_dim = len(next(iter(embeddings.values())))

    # --- Exact match ---
    if word in embeddings:
        return np.asarray(embeddings[word], dtype=np.float32)

    # --- Compound word handling ---
    if "_" in word or "-" in word:
        parts = word.replace("-", "_").split("_")
        part_embeddings = []

        for p in parts:
            if p in embeddings:
                part_embeddings.append(
                    np.asarray(embeddings[p], dtype=np.float32)
                )
            else:
                stem = stemmer.stem(p)
                stem_related = [
                    v for k, v in embeddings.items()
                    if stemmer.stem(k) == stem
                ]
                if stem_related:
                    arr = np.stack(stem_related, axis=0).astype(np.float32)
                    part_embeddings.append(arr.mean(axis=0))

        if part_embeddings:
            return np.mean(np.stack(part_embeddings, axis=0), axis=0)

    # --- Single-word stem fallback ---
    stem = stemmer.stem(word)
    stem_related = [
        v for k, v in embeddings.items()
        if stemmer.stem(k) == stem
    ]
    if stem_related:
        arr = np.stack(stem_related, axis=0).astype(np.float32)
        return arr.mean(axis=0)

    # --- Fallback ---
    print("Failed word: " + str(word) + "\n")
    return None


def print_pairwise_statistics(df, response_col, comparison_col, label, model):
    response = df[response_col]
    comparison = df[comparison_col]

    print(f"\nStatistics: {label} for {model}")
    print(f"  Mann–Whitney p-value: {get_mannwhitney_p_value(response, comparison):.4f}")
    print(f"  t-test p-value:       {get_t_p_value(response, comparison):.4f}")
    
    spearman_corr, spearman_p = get_spearman_corr(response, comparison)
    print(f"  Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.4f})")
    
    pearson_corr, pearson_p = get_pearson_corr(response, comparison)
    print(f"  Pearson correlation:  {pearson_corr:.4f} (p={pearson_p:.4f})")
    
    print(f"  Kolmogorov–Smirnov p-value: {get_ks_p_value(response, comparison):.2e}")


def parse_box(response: str):
    match = re.findall(r'(/boxed\[|/boxed |/boxed<|/boxed{|\\boxed\[|\\boxed<|\\boxed{)(.*?)(}|>|\]|}\\|>\\|\]\\|\))', response)
    try:
        return match[-1][1].lower()
    except:
        sliced = response
        if "</think>" in sliced: #for reasoning models
            sliced = sliced[sliced.find("</think>") + len("</think>"):]
        try:
            match = re.findall(r'(/boxed\[|/boxed<|/boxed{|\\boxed\[|\\boxed<|\\boxed{)(.*?)(}|>|\]|}\\|>\\|\]\\)', response)
            return match[-1][1].lower()
        except:
            return "E"

def get_embeddings_for_each_word(path= home + "/MetaphorReasoning/questions/results/",
    files= ["open_target/deepseek-R1-target-open.csv", 
            "open_target/gpt-4o-target-open.csv",
            "open_target/gemma-27b-target-open.csv"], is_cot= False):
    embeddings = get_conceptnet_embeddings()
    global_data = pd.read_csv("data_with_target_specificity_scores.csv")
    global_data = global_data.reset_index(drop= False)
    files = [path + "/" + f for f in files]
    i = 0
    for file in files:
        file_df = pd.read_csv(file).reset_index(drop= False)
        merged = global_data.merge(file_df, left_index= True, right_index= True, how= "inner")
        merged.dropna(inplace= True) #same order, no big deal
        response_lexeme_sims = []
        response_paraphrase_sims = []
        for index, row in merged.iterrows():
            answer = merged.at[index, "full_answer"]
            target_lexeme = merged.at[index, "original_target"]
            literal_paraphrase = merged.at[index, "noun_target"]
            parsed = parse_box(answer)
            if(parsed == "E"):
                print("couldn't parse answer...\n")
                response_lexeme_sims.append(float('NaN'))
                response_paraphrase_sims.append(float('NaN'))
            else:
                tagged = tag_target(parsed)
                noun = extract_noun(tagged)
                if(noun == ""):
                    print("No noun detected\n")
                    response_lexeme_sims.append(float('NaN'))
                    response_paraphrase_sims.append(float('NaN'))
                else:
                    answer_embedding = get_features(noun, embeddings)
                    lexeme_embedding = get_features(target_lexeme, embeddings)
                    literal_embedding = get_features(literal_paraphrase, embeddings)
                    if(answer_embedding is None or lexeme_embedding is None or literal_embedding is None):
                        print("Failed to get an embedding...\n")
                        response_lexeme_sims.append(float('NaN'))
                        response_paraphrase_sims.append(float('NaN'))
                    else:
                        response_lexeme_sims.append(cos_sim(answer_embedding, lexeme_embedding))
                        response_paraphrase_sims.append(cos_sim(answer_embedding, literal_embedding))
        #print("Done printing iterrows indicies...\n")
        results_df = pd.DataFrame({
            "response_target_lexeme_score": response_lexeme_sims,
            "response_literal_paraphrase_score": response_paraphrase_sims,
        })
        results_df.dropna(inplace= True)
        model = file[file.rindex("/"):file.index("-target")]
        if(is_cot):
            model = model + "_cot_"
        graph_data(
            results_df,
            f"{model}_cos_sim_open_results"
        )
        results_df.to_csv("data/" + model + "_cos_sims.csv", index= False)
        i += 1

#new_input_files = ["deepseek-R1-target-cot_open.csv", 
            #"gpt-4o-target-cot_open.csv",
            #"gemma-27b-target-cot_open.csv"]

#get_embeddings_for_each_word(files= new_input_files, is_cot= True)
#print("Done...\n")
#time.sleep(60)

def calcualte_score_distribution(data) -> tuple:
    score_1s = []
    score_3s = []
    for index, row in data.iterrows():
        answer = data.at[index, "full_answer"]
        parsed = parse_box(answer)
        if(parsed == "E"):
            print("Invalid bracketing\n")
            score_1s.append(float("nan"))
            score_3s.append(float("nan"))
        else:
            tagged = tag_target(parsed)
            noun = extract_noun(tagged)
            if(noun == ""):
                print("No noun detected\n")
                score_1s.append(float("nan"))
                score_3s.append(float("nan"))
            else:
                score_1s.append(get_first_score(noun))
                score_3s.append(get_third_score(noun))
    return score_1s, score_3s

def create_results_csvs():
    directory = home + '/MetaphorReasoning/questions/results'
    extension = "target-open.csv"
    for file in os.listdir(directory):
        if file.endswith(extension):
            results_df = pd.DataFrame(columns= [
                'model',
                'response_score_1',
                'response_score_3'
            ])
            df = pd.read_csv(os.path.join(directory, file))
            model_name = file[:file.index("-target")]
            results_df["model"] = model_name
            response_scores_1, response_scores_3 = calcualte_score_distribution(df)
            results_df["response_score_1"] = response_scores_1
            results_df["response_score_3"] = response_scores_3
            output_file = os.path.join(directory + "/summary", f"{model_name}_open_results_summary.csv")
            results_df.to_csv(output_file, index= False)

#print("Done...\n")
#time.sleep(60)

scaler = MinMaxScaler(feature_range=(0, 1))


def create_comparision_file(path= home + "/MetaphorReasoning/questions/results/summary", 
    files= ["deepseek-R1_open_results_summary.csv", 
            "gpt-4o_open_results_summary.csv",
            "gemma-27b_open_results_summary.csv"]):
    global_data = pd.read_csv("data_with_target_specificity_scores.csv")
    global_data = global_data.reset_index(drop= False)
    files = [path + "/" + f for f in files]
    
    all_model_dfs = []
    summary_rows = []
    for file in files:
        results_df = pd.DataFrame(columns= [
            'model',
            'response_score_1',
            'response_score_2',
            'dataset_target_score_1',
            'dataset_target_score_2',
            'specific_target_score_1',
            'specific_target_score_2'
        ])
        file_df = pd.read_csv(file).reset_index(drop= False)
        merged = global_data.merge(file_df, left_index= True, right_index= True, how= "inner")
        merged = merged.drop(columns= ['model'])
        merged.dropna(inplace= True) #same order, no big deal
        results_df["response_score_1"] = merged["response_score_1"]
        results_df["response_score_2"] = merged["response_score_3"]
        results_df["dataset_target_score_1"] = merged["original_target_score_1"]
        results_df["dataset_target_score_2"] = merged["original_target_score_3"]
        results_df["specific_target_score_1"] = merged["noun_target_score_1"]
        results_df["specific_target_score_2"] = merged["noun_target_score_3"]
        score_columns = [
            "response_score_1",
            "response_score_2",
            "dataset_target_score_1",
            "dataset_target_score_2",
            "specific_target_score_1",
            "specific_target_score_2",
        ]
        results_df[score_columns] = scaler.fit_transform(results_df[score_columns])
        model = file[file.index(""):file.index("_open")]

        # additions: track model and store df
        results_df["model"] = model
        model_df = results_df.copy()
        model_df["row_id"] = model_df.index
        all_model_dfs.append(model_df)

        # ---- Score 1 ----
        print_pairwise_statistics(
            results_df,
            response_col= "response_score_1",
            comparison_col= "dataset_target_score_1",
            label= "response_score_1 vs dataset_target_score_1",
            model= model
        )

        print_pairwise_statistics(
            results_df,
            response_col= "response_score_1",
            comparison_col= "specific_target_score_1",
            label= "response_score_1 vs specific_target_score_1",
            model= model
        )

        # ---- Score 2 ----
        print_pairwise_statistics(
            results_df,
            response_col= "response_score_2",
            comparison_col= "dataset_target_score_2",
            label= "response_score_2 vs dataset_target_score_2",
            model= model
        )

        print_pairwise_statistics(
            results_df,
            response_col= "response_score_2",
            comparison_col= "specific_target_score_2",
            label= "response_score_2 vs specific_target_score_2",
            model= model
        )

        #additions: compute mean and sd for summary
        means = results_df[score_columns].mean()
        sds = results_df[score_columns].std()

        summary_row = {"model": model}
        for col in score_columns:
            summary_row[f"{col}_mean"] = means[col]
            summary_row[f"{col}_sd"] = sds[col]

        summary_rows.append(summary_row)

    #additions after loop
    merged_models = all_model_dfs[0]
    for df in all_model_dfs[1:]:
        merged_models = merged_models.merge(
            df,
            on= "row_id",
            how= "inner",
            suffixes=("", "_dup")
        )

    clean_model_dfs = []
    for df in all_model_dfs:
        clean_df = df[df["row_id"].isin(merged_models["row_id"])].copy()
        clean_model_dfs.append(clean_df)

    summary_df = pd.DataFrame(summary_rows)
    print("n_samples_shared_across_models:", len(merged_models))

    print(summary_df)

    summary_df.to_csv("statistics/open_ended_target_specificity_summary.csv", index= False)

if __name__ == "__main__":
    create_results_csvs()
    create_comparision_file()
    print(f"Completed building open ended target specificity summary file in directory: statistics/open_ended_target_specificity_summary.csv...\n")