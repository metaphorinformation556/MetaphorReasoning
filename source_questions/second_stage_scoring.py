'''
This is the file for implementing and saving metrics for the second stage task.

The following metrics will be saved to a csv in the 'results' folder:
    - Percentage of samples where model said the word was metaphorical (and people didn't) but in the second stage said it was literal
    - Percentage of samples where at least one instance is downweighted by the model
    - Percentage of samples where at least one instance is upweighted by the model
    - Percentage of samples where people say the word is metaphorical but the model says it is literal
    - Percentage of samples where people say the word is literal but the model says it is metaphorical
    - Total distribution of scores as a percentage of each metaphoricity score category

Exclusion criteria:
    - All samples where the model returned an invalid dictionary
    - All samples where annotators marked a word as grammatically invalid
    - All samples with more than one annotation for a single word
    - All samples with more than one instance of the same word
'''

import pandas as pd
import ast
from second_stage_metrics import (
    parse_response,
    create_human_ratings,
    get_human_ratings,
    filter_human_ratings,
    get_llm_ratings_of_overlapped_samples,
    get_llm_ratings_of_overlapped_samples_,
    get_llm_ratings_of_llm_only_samples,
    self_inconsistancy_score,
    self_inconsistancy_bool,
    human_contradiction_scores,
    human_contradiction_scores_bool,
)
import time

def break_dictionary_up(input: dict) -> list[int]:
    input = filter_human_ratings(input) #eliminate invalid scores
    return [len(input[key]) for key in input.keys()]

if __name__ == "__main__":
    deepseek_data = pd.read_csv("results/open_source/2nd_stage_deepseek-R1.csv")
    gpt_data = pd.read_csv("results/open_source/2nd_stage_gpt_4o.csv")

    deepseek_data["lm_source"] = deepseek_data["lm_source"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    deepseek_data["metaphoricity_score"] = deepseek_data["metaphoricity_score"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    deepseek_data["parsed_answer"] = deepseek_data["parsed_answer"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    deepseek_data["human_and_gpt"] = deepseek_data["human_and_gpt"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    deepseek_data["llm_only"] = deepseek_data["llm_only"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    deepseek_data["all_overlap"] = deepseek_data["all_overlap"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    gpt_data["lm_source"] = gpt_data["lm_source"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    gpt_data["metaphoricity_score"] = gpt_data["metaphoricity_score"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    gpt_data["parsed_answer"] = gpt_data["parsed_answer"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    gpt_data["human_and_gpt"] = gpt_data["human_and_gpt"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    gpt_data["llm_only"] = gpt_data["llm_only"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    gpt_data["all_overlap"] = gpt_data["all_overlap"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    deepseek_data["is_mistake"] = deepseek_data["metaphoricity_score"].apply(lambda x: True if (len(x) == 1 and x[0] == -1.0) else False) #filter those annotated as invalid
    gpt_data["is_mistake"] = gpt_data["metaphoricity_score"].apply(lambda x: True if (len(x) == 1 and x[0] == -1.0) else False)

    deepseek_data = deepseek_data[deepseek_data["is_mistake"] == False]
    gpt_data = gpt_data[gpt_data["is_mistake"] == False]

    assert(len(deepseek_data) == len(gpt_data))

    print(f"There are {len(deepseek_data)} samples.\n")

    deepseek_data["response_dictionary"] = deepseek_data["full_answer"].apply(lambda x: parse_response(x))
    gpt_data["response_dictionary"] = gpt_data["full_answer"].apply(lambda x: parse_response(x))
    
    deepseek_data.dropna(inplace= True)
    gpt_data.dropna(inplace= True)

    print(f"After dropping invalid responses, there are {len(deepseek_data)} deepseek samples and {len(gpt_data)} gpt samples.\n")

    deepseek_data["human_dictionary"] = deepseek_data.apply(lambda x: create_human_ratings(x["lm_source"], x["metaphoricity_score"]), axis= 1)
    gpt_data["human_dictionary"] = gpt_data.apply(lambda x: create_human_ratings(x["lm_source"], x["metaphoricity_score"]), axis= 1)

    deepseek_data["llm_only_samples"] = deepseek_data.apply(lambda x: get_llm_ratings_of_llm_only_samples(x["human_dictionary"], x["response_dictionary"]), axis= 1)
    gpt_data["llm_only_samples"] = gpt_data.apply(lambda x: get_llm_ratings_of_llm_only_samples(x["human_dictionary"], x["response_dictionary"]), axis= 1)
    deepseek_data["human_only_samples"] = deepseek_data["human_dictionary"].apply(lambda x: get_human_ratings(x))
    gpt_data["human_only_samples"] = gpt_data["human_dictionary"].apply(lambda x: get_human_ratings(x))

    deepseek_data.dropna(inplace= True)
    gpt_data.dropna(inplace= True)

    print(f"After dropping invalid responses, there are {len(deepseek_data)} deepseek samples and {len(gpt_data)} gpt samples.\n")

    deepseek_data["is_inconsistancy"] = deepseek_data.apply(lambda x: self_inconsistancy_bool(x["human_dictionary"], x["response_dictionary"]), axis= 1) #for first metric to be reported
    gpt_data["is_inconsistancy"] = gpt_data.apply(lambda x: self_inconsistancy_bool(x["human_dictionary"], x["response_dictionary"]), axis= 1)

    deepseek_data.dropna(inplace= True)
    gpt_data.dropna(inplace= True)

    print(f"After dropping invalid responses, there are {len(deepseek_data)} deepseek samples and {len(gpt_data)} gpt samples.\n")
    
    deepseek_data[["is_more", "is_less", "is_equal", "is_literal", "is_metaphorical"]] = deepseek_data.apply(lambda x: 
        pd.Series(human_contradiction_scores_bool(x["human_dictionary"], x["response_dictionary"])), axis= 1) #for metrics 2-5
    gpt_data[["is_more", "is_less", "is_equal", "is_literal", "is_metaphorical"]] = gpt_data.apply(lambda x: 
        pd.Series(human_contradiction_scores_bool(x["human_dictionary"], x["response_dictionary"])), axis= 1)
    
    deepseek_data.dropna(inplace= True)
    gpt_data.dropna(inplace= True)

    print(f"After dropping invalid ratings, there are {len(deepseek_data)} deepseek samples and {len(gpt_data)} gpt samples.\n")

    deepseek_data[["human_score_0", "human_score_1", "human_score_2", "human_score_3"]] = deepseek_data.apply(lambda x: 
        pd.Series(break_dictionary_up(x["human_dictionary"])), axis= 1) #for rest of metrics
    gpt_data[["human_score_0", "human_score_1", "human_score_2", "human_score_3"]] = gpt_data.apply(lambda x: 
        pd.Series(break_dictionary_up(x["human_dictionary"])), axis= 1) #for rest of metrics

    deepseek_data[["llm_score_0", "llm_score_1", "llm_score_2", "llm_score_3"]] = deepseek_data.apply(lambda x: 
        pd.Series(break_dictionary_up(x["response_dictionary"])), axis= 1) #for rest of metrics
    gpt_data[["llm_score_0", "llm_score_1", "llm_score_2", "llm_score_3"]] = gpt_data.apply(lambda x: 
        pd.Series(break_dictionary_up(x["response_dictionary"])), axis= 1) #for rest of metrics
    
    deepseek_data[["llm_human_only_score_0", "llm_human_only_score_1", "llm_human_only_score_2", "llm_human_only_score_3"]] = deepseek_data.apply(
        lambda x: break_dictionary_up(
            get_llm_ratings_of_overlapped_samples_(x["human_dictionary"], x["response_dictionary"])
        ),
        axis= 1,
        result_type= "expand"
    )

    gpt_data[["llm_human_only_score_0", "llm_human_only_score_1", "llm_human_only_score_2", "llm_human_only_score_3"]] = gpt_data.apply(
        lambda x: break_dictionary_up(
            get_llm_ratings_of_overlapped_samples_(x["human_dictionary"], x["response_dictionary"])
        ),
        axis= 1,
        result_type= "expand"
    )

    deepseek_data.dropna(inplace= True)
    gpt_data.dropna(inplace= True)

    print(f"After dropping invalid ratings, there are {len(deepseek_data)} deepseek samples and {len(gpt_data)} gpt samples.\n")

    deepseek_data.to_csv("results/open_source/2nd_stage_deepseek_cleaned.csv", index= False)
    gpt_data.to_csv("results/open_source/2nd_stage_gpt_cleaned.csv", index= False)
    
    deepseek_data["total_human_tokens"] = deepseek_data["lm_source"].apply(lambda x: len(x))
    gpt_data["total_human_tokens"] = gpt_data["lm_source"].apply(lambda x: len(x))

    deepseek_data["total_llm_tokens"] = deepseek_data["all_overlap"].apply(lambda x: len(x))
    gpt_data["total_llm_tokens"] = gpt_data["all_overlap"].apply(lambda x: len(x))

    deepseek_data["total_overlap"] = deepseek_data["human_and_gpt"].apply(lambda x: len(x))
    gpt_data["total_overlap"] = gpt_data["human_and_gpt"].apply(lambda x: len(x))

    '''Actually calculating metrics for the final csv'''

    deepseek_llm_inconsistancy_score = deepseek_data["is_inconsistancy"].sum() / len(deepseek_data)
    gpt_llm_inconsistancy_score = gpt_data["is_inconsistancy"].sum() / len(gpt_data)

    deepseek_more_score = deepseek_data["is_more"].sum() / len(deepseek_data)
    deepseek_less_score = deepseek_data["is_less"].sum() / len(deepseek_data)
    deepseek_equal_score = deepseek_data["is_equal"].sum() / len(deepseek_data)
    deepseek_literal_score = deepseek_data["is_literal"].sum() / len(deepseek_data)
    deepseek_metaphorical_score = deepseek_data["is_metaphorical"].sum() / len(deepseek_data)

    gpt_more_score = gpt_data["is_more"].sum() / len(gpt_data)
    gpt_less_score = gpt_data["is_less"].sum() / len(gpt_data)
    gpt_equal_score = gpt_data["is_equal"].sum() / len(gpt_data)
    gpt_literal_score = gpt_data["is_literal"].sum() / len(gpt_data)
    gpt_metaphorical_score = gpt_data["is_metaphorical"].sum() / len(gpt_data)

    deepseek_total_human_tokens = deepseek_data["total_human_tokens"].sum()
    gpt_total_human_tokens = gpt_data["total_human_tokens"].sum()

    deepseek_total_llm_tokens = deepseek_data["total_llm_tokens"].sum()
    gpt_total_llm_tokens = gpt_data["total_llm_tokens"].sum()

    deepseek_total_human_only_llm_tokens = deepseek_data["total_overlap"].sum()
    gpt_total_human_only_llm_tokens = gpt_data["total_overlap"].sum()

    deepseek_human_score_0 = deepseek_data["human_score_0"].sum() / deepseek_total_human_tokens
    deepseek_human_score_1 = deepseek_data["human_score_1"].sum() / deepseek_total_human_tokens
    deepseek_human_score_2 = deepseek_data["human_score_2"].sum() / deepseek_total_human_tokens
    deepseek_human_score_3 = deepseek_data["human_score_3"].sum() / deepseek_total_human_tokens

    gpt_human_score_0 = gpt_data["human_score_0"].sum() / gpt_total_human_tokens
    gpt_human_score_1 = gpt_data["human_score_1"].sum() / gpt_total_human_tokens
    gpt_human_score_2 = gpt_data["human_score_2"].sum() / gpt_total_human_tokens
    gpt_human_score_3 = gpt_data["human_score_3"].sum() / gpt_total_human_tokens

    deepseek_llm_score_0 = deepseek_data["llm_score_0"].sum() / deepseek_total_llm_tokens
    deepseek_llm_score_1 = deepseek_data["llm_score_1"].sum() / deepseek_total_llm_tokens
    deepseek_llm_score_2 = deepseek_data["llm_score_2"].sum() / deepseek_total_llm_tokens
    deepseek_llm_score_3 = deepseek_data["llm_score_3"].sum() / deepseek_total_llm_tokens

    gpt_llm_score_0 = gpt_data["llm_score_0"].sum() / gpt_total_llm_tokens
    gpt_llm_score_1 = gpt_data["llm_score_1"].sum() / gpt_total_llm_tokens
    gpt_llm_score_2 = gpt_data["llm_score_2"].sum() / gpt_total_llm_tokens
    gpt_llm_score_3 = gpt_data["llm_score_3"].sum() / gpt_total_llm_tokens

    deepseek_human_only_llm_score_0 = deepseek_data["llm_human_only_score_0"].sum() / deepseek_total_human_tokens
    deepseek_human_only_llm_score_1 = deepseek_data["llm_human_only_score_1"].sum() / deepseek_total_human_tokens
    deepseek_human_only_llm_score_2 = deepseek_data["llm_human_only_score_2"].sum() / deepseek_total_human_tokens
    deepseek_human_only_llm_score_3 = deepseek_data["llm_human_only_score_3"].sum() / deepseek_total_human_tokens

    gpt_human_only_llm_score_0 = gpt_data["llm_human_only_score_0"].sum() / gpt_total_human_tokens
    gpt_human_only_llm_score_1 = gpt_data["llm_human_only_score_1"].sum() / gpt_total_human_tokens
    gpt_human_only_llm_score_2 = gpt_data["llm_human_only_score_2"].sum() / gpt_total_human_tokens
    gpt_human_only_llm_score_3 = gpt_data["llm_human_only_score_3"].sum() / gpt_total_human_tokens


    final_metrics = pd.DataFrame([{
        "model": "deepseek",
        "llm_inconsistency": deepseek_llm_inconsistancy_score,
        "more": deepseek_more_score,
        "less": deepseek_less_score,
        "equal": deepseek_equal_score,
        "literal": deepseek_literal_score,
        "metaphorical": deepseek_metaphorical_score,
        "human_score_0": deepseek_human_score_0,
        "human_score_1": deepseek_human_score_1,
        "human_score_2": deepseek_human_score_2,
        "human_score_3": deepseek_human_score_3,
        "llm_score_0": deepseek_llm_score_0,
        "llm_score_1": deepseek_llm_score_1,
        "llm_score_2": deepseek_llm_score_2,
        "llm_score_3": deepseek_llm_score_3,
        "human_only_llm_score_0": deepseek_human_only_llm_score_0,
        "human_only_llm_score_1": deepseek_human_only_llm_score_1,
        "human_only_llm_score_2": deepseek_human_only_llm_score_2,
        "human_only_llm_score_3": deepseek_human_only_llm_score_3,
    }, {
        "model": "gpt",
        "llm_inconsistency": gpt_llm_inconsistancy_score,
        "more": gpt_more_score,
        "less": gpt_less_score,
        "equal": gpt_equal_score,
        "literal": gpt_literal_score,
        "metaphorical": gpt_metaphorical_score,
        "human_score_0": gpt_human_score_0,
        "human_score_1": gpt_human_score_1,
        "human_score_2": gpt_human_score_2,
        "human_score_3": gpt_human_score_3,
        "llm_score_0": gpt_llm_score_0,
        "llm_score_1": gpt_llm_score_1,
        "llm_score_2": gpt_llm_score_2,
        "llm_score_3": gpt_llm_score_3,
        "human_only_llm_score_0": gpt_human_only_llm_score_0,
        "human_only_llm_score_1": gpt_human_only_llm_score_1,
        "human_only_llm_score_2": gpt_human_only_llm_score_2,
        "human_only_llm_score_3": gpt_human_only_llm_score_3,
    }])

    final_metrics.to_csv("metrics/second_stage_final_metrics.csv", index= False)

    print("Saved metrics to directory: metrics/second_stage_final_metrics.csv\n")




