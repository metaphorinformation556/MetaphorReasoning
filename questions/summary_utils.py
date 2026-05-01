import argparse
import pandas as pd
import os
import ast
import re
from pathlib import Path

parser = argparse.ArgumentParser(
    prog='RunQuestions',
    description='Generates multiple choice questions.')

parser.add_argument('--type', type= str, choices= [
    'open', 'open_cot', 'mcq_2', 'mcq_4', 'mcq_none_or_all',
    'open_source', 'open_source_stage_2',
    'v_source', 'n_source',
    'antonym_original_target', 'antonym_our_target',
    'baseline_our_target', 'baseline_original_target',
    'pseudoword_our_target', 'pseudoword_original_target',
])

args = parser.parse_args()

def extract_response(response: str) -> str:
    # 2. Primary: boxed answers
    boxed_pattern = re.compile(
        r'(?:/boxed\[|/boxed<|/boxed{|\\boxed\[|\\boxed<|\\boxed{)\s*([A-Ea-e])',
        re.DOTALL
    )
    boxed_matches = boxed_pattern.findall(response)
    if boxed_matches:
        return boxed_matches[-1].upper()

    # 3. Secondary: explicit answer markers
    marker_pattern = re.compile(
        r'(?:answer:|final answer|Answer:|ANSWER:|\*\*Answer\*\*|\*Answer\*)\s*([A-Ea-e])',
        re.IGNORECASE
    )
    marker_match = marker_pattern.search(response)
    if marker_match:
        return marker_match.group(1).upper()
    
     # 1. Strip reasoning section if present
    if "</think>" in response:
        response = response.split("</think>", 1)[1]
        boxed_matches = boxed_pattern.findall(response)
        if boxed_matches:
            return boxed_matches[-1].upper()

        marker_match = marker_pattern.search(response)
        if marker_match:
            return marker_match.group(1).upper()
        
    #Default
    return "E"

def fix_original(permutation, old_dictionary):
    new_dictionary = {}
    old_keys = list(old_dictionary.keys())
    permutation_values = list(permutation.values())
    new_dictionary[old_keys[0]] = "A"
    new_dictionary[old_keys[1]] = "B"
    third_key = ""
    for value in permutation_values:
        if("Both options" in value):
            third_key = value
            break
    new_dictionary[third_key] = "C"
    new_dictionary["None of the options"] = "D"
    return new_dictionary

def calculate_score_distribution(data: pd.DataFrame, type: str) -> dict:
    data["permutation"] = data["permutation"].apply(lambda x: ast.literal_eval(x))
    data["original"] = data["original"].apply(lambda x: ast.literal_eval(x))
    if(type == "mcq_none_or_all"): #error with original dictionary
        data["original"] = data.apply(lambda x: fix_original(x["permutation"], x["original"]), axis= 1)
    if(type == "mcq_2"):
        results = {"A": 0, "B": 0, "E": 0}
        for index, row in data.iterrows():
            full_answer = data.at[index, "full_answer"]
            permuted = extract_response(full_answer)
            permutation_dictionary = data.at[index, "permutation"]
            original_dictionary = data.at[index, "original"]
            try:
                answer = original_dictionary[permutation_dictionary[permuted]]
            except:
                answer = "E"
            if(answer == "A"):
                results["A"] += 1
            elif(answer == "B"):
                results["B"] += 1
            else:
                results["E"] += 1
        for key in results:
            curr = results[key]
            results[key] = curr/len(data)
        return results
    elif (type == "mcq_4" or type == "mcq_none_or_all"):
        results = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
        for index, row in data.iterrows():
            full_answer = data.at[index, "shuffled_answer"]
            permuted = full_answer
            #permuted = extract_response(full_answer)
            permutation_dictionary = data.at[index, "permutation"]
            original_dictionary = data.at[index, "original"]
            try:
                answer = original_dictionary[permutation_dictionary[permuted]]
            except:
                answer = "E"
            if(answer == "A"):
                results["A"] += 1
            elif(answer == "B"):
                results["B"] += 1
            elif(answer == "C"):
                results["C"] += 1
            elif(answer == "D"):
                results["D"] += 1
            else:
                results["E"] += 1
        for key in results:
            curr = results[key]
            results[key] = curr/len(data)
        return results

#assume mapping baseline, so correct answer is A
def calculate_if_correct(letter: str, permutation: dict, original: dict):
    try:
        answer = original[permutation[letter]]
    except:
        answer = "E"
    if(answer == "A"):
        return 1
    else:
        return 0

def create_target_results_csv(type: str):
    if(type == "mcq_2"):
        results_df = pd.DataFrame(columns= ['model', 'A', 'B', 'E'])
    else:
        results_df = pd.DataFrame(columns= ['model', 'A', 'B', 'C', 'D', 'E'])
    if(type == 'mcq_2'):
        directory = f'results/mcq_target/2_option'
    elif(type == 'mcq_4' or type == 'mcq_none_or_all'):
        directory = f"results/mcq_target/4_option"
    for file in os.listdir(directory):
        if type in file and "few_shot" not in file:
            if type == "mcq_2":
                results_series = pd.Series({
                    'model': None,
                    'A': None,
                    'B': None,
                    'E': None
                })
            elif(type == "mcq_4" or type == "mcq_none_or_all"):
                results_series = pd.Series({
                    'model': None,
                    'A': None,
                    'B': None,
                    'C': None,
                    'D': None,
                    'E': None
                })
            df = pd.read_csv(os.path.join(directory, file))
            df["shuffled_answer"] = df["full_answer"].apply(extract_response)
            model_name = file[:file.index(f"-target-{type}")]
            results_series["model"] = model_name
            score_dist_dictionary = calculate_score_distribution(df, type)
            if(type == "mcq_2"):
                results_series["A"] = score_dist_dictionary["A"]
                results_series["B"] = score_dist_dictionary["B"]
                results_series["E"] = score_dist_dictionary["E"]
            else:
                results_series["A"] = score_dist_dictionary["A"]
                results_series["B"] = score_dist_dictionary["B"]
                results_series["C"] = score_dist_dictionary["C"]
                results_series["D"] = score_dist_dictionary["D"]
                results_series["E"] = score_dist_dictionary["E"]
            results_df = pd.concat([results_df, results_series.to_frame().T], ignore_index= True)
    output_file = os.path.join(directory + "/summary", f"{type}_results_summary.csv")
    results_df.to_csv(output_file, index= False)

def calculate_score_distribution_source(data: pd.DataFrame) -> dict:
    data["permutation"] = data["permutation"].apply(lambda x: ast.literal_eval(x))
    data["original"] = data["original"].apply(lambda x: ast.literal_eval(x))
    results = {"A": 0, "B": 0, "C": 0, "E": 0}
    for index, row in data.iterrows():
        answer = data.at[index, "shuffled_answer"]
        permutation_dictionary = data.at[index, "permutation"]
        original_dictionary = data.at[index, "original"]
        try:
            answer = original_dictionary[permutation_dictionary[answer]]
        except:
            answer = "E"
        if(answer == "A"):
            results["A"] += 1
        elif(answer == "B"):
            results["B"] += 1
        elif(answer == "C"):
            results["C"] += 1
        else:
            results["E"] += 1
    for key in results:
        curr = results[key]
        results[key] = curr/len(data)
    return results

def create_source_results_csv(type: str):
    results_df = pd.DataFrame(columns= ['model', 'A', 'B', 'C', 'E'])
    directory = 'results/mcq_source/'
    file_prefix = {"v_source": "vanilla_source", "n_source": "normal_source"}.get(type, type)
    for file in os.listdir(directory):
        if file_prefix in file:
            results_series = pd.Series({
                'model': None,
                'A': None,
                'B': None,
                'C': None,
                'E': None
            })
            df = pd.read_csv(os.path.join(directory, file))
            df["shuffled_answer"] = df["full_answer"].apply(extract_response)
            model_name = file[file.rindex("_") + 1:file.index(".csv")]
            if("gemma" in model_name or "qwen" in model_name or "deepseek-R1" in model_name or "gpt-4o" in model_name):
                home = Path.home()
                if type == "v_source" or "vanilla" in type:
                    to_merge = pd.read_csv(home / "MetaphorMemorizationOrReasoning/source_questions/data/updated_mcq_vanilla_source_questions.csv")
                else:
                    to_merge = pd.read_csv(home / "MetaphorMemorizationOrReasoning/source_questions/data/updated_mcq_normal_source_questions.csv")
                to_merge = to_merge[["normal_question", "correct_letter", "original", "permutation"]]
                df = df.merge(to_merge, left_on= "normal_question", right_on= "normal_question")
            model_name = file[file.rindex("_"):file.index(".csv")]
            results_series["model"] = model_name
            score_dist_dictionary = calculate_score_distribution_source(df)
            results_series["A"] = score_dist_dictionary["A"]
            results_series["B"] = score_dist_dictionary["B"]
            results_series["C"] = score_dist_dictionary["C"]
            results_series["E"] = score_dist_dictionary["E"]
            results_df = pd.concat([results_df, results_series.to_frame().T], ignore_index= True)
    output_file = os.path.join(directory + "/summary", f"{type}_results_summary.csv")
    results_df.to_csv(output_file, index= False)

def calculate_score_distribution_mapping(data: pd.DataFrame, type: str) -> dict:
    data["permutation"] = data["permutation"].apply(lambda x: ast.literal_eval(x))
    data["original"] = data["original"].apply(lambda x: ast.literal_eval(x))
    if (type == "antonym_original_target" or type == "antonym_no_target"
                 or type == "antonym_our_target" or type == "baseline_no_target" or type == "baseline_our_target"
                 or type == "baseline_original_target" or type == "pseudoword_no_target" or type == "pseudoword_our_target"
                 or type == "pseudoword_original_target"):
        results = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
        for index, row in data.iterrows():
            full_answer = data.at[index, "shuffled_answer"]
            permuted = full_answer
            #permuted = extract_response(full_answer)
            permutation_dictionary = data.at[index, "permutation"]
            original_dictionary = data.at[index, "original"]
            try:
                answer = original_dictionary[permutation_dictionary[permuted]]
            except:
                answer = "E"
            if(answer == "A"):
                results["A"] += 1
            elif(answer == "B"):
                results["B"] += 1
            elif(answer == "C"):
                results["C"] += 1
            elif(answer == "D"):
                results["D"] += 1
            else:
                results["E"] += 1
        for key in results:
            curr = results[key]
            results[key] = curr/len(data)
        return results

def create_mapping_results_csv(type: str):
    results_df = pd.DataFrame(columns= ['model', 'A', 'B', 'C', 'D', 'E'])
    directory = 'results/mapping'
    for file in os.listdir(directory):
        if type in file and "few_shot" not in file and "final" in file:
            if(type == "antonym_original_target" or type == "antonym_no_target"
                 or type == "antonym_our_target" or type == "baseline_no_target" or type == "baseline_our_target"
                 or type == "baseline_original_target" or type == "pseudoword_no_target" or type == "pseudoword_our_target"
                 or type == "pseudoword_original_target"):
                results_series = pd.Series({
                    'model': None,
                    'A': None,
                    'B': None,
                    'C': None,
                    'D': None,
                    'E': None
                })
            df = pd.read_csv(os.path.join(directory, file))
            df["shuffled_answer"] = df["full_answer"].apply(extract_response)
            model_name = file[file.rindex("_") + 1:file.index(".csv")]
            if("gemma" in model_name or "qwen" in model_name):
                to_merge = pd.read_csv(f"mapping_data/for_llms/final_{type}.csv")
                to_merge = to_merge[["normal_question", "correct_letter", "original", "permutation"]]
                df = df.merge(to_merge, left_on= "normal_question", right_on= "normal_question")
            model_name = file[file.rindex("_"):file.index(".csv")]
            results_series["model"] = model_name
            score_dist_dictionary = calculate_score_distribution_mapping(df, type)
            results_series["A"] = score_dist_dictionary["A"]
            results_series["B"] = score_dist_dictionary["B"]
            results_series["C"] = score_dist_dictionary["C"]
            results_series["D"] = score_dist_dictionary["D"]
            results_series["E"] = score_dist_dictionary["E"]
            results_df = pd.concat([results_df, results_series.to_frame().T], ignore_index= True)
    output_file = os.path.join(directory + "/summary", f"{type}_results_summary.csv")
    results_df.to_csv(output_file, index= False)

if args.type in ["mcq_2", "mcq_4", "mcq_none_or_all"]:
    create_target_results_csv(args.type)
    if(args.type == "mcq_2"):
        print(f"results saved to directory: results/mcq_target/{args.type}/summary\n")
    else:
        print(f"results saved to directory: results/mcq_source/mcq_4/summary\n")
elif args.type in ["v_source", "n_source"]:
    create_source_results_csv(args.type)
    print(f"results saved to directory: results/mcq_source/summary\n")
elif args.type in [
    "antonym_original_target", "antonym_our_target",
    "baseline_our_target", "baseline_original_target",
    "pseudoword_our_target", "pseudoword_original_target",
]:
    create_mapping_results_csv(args.type)
    print(f"results saved to directory: results/mapping/summary\n")