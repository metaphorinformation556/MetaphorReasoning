import pandas as pd
import re
from pathlib import Path
import numpy as np
import pickle
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

home = str(Path.home())

PATH = home + "/metaphor_project_refactor/metaphor_project/questions/results/open_target/"

conceptnet_path = home + "/advertising/GNN/conceptnet_embeddings/embeddings.pkl"

files = [
    "deepseek-R1-target-open.csv",
    "gpt-4o-target-open.csv",
]

#get dict of conceptnet embeddings
def get_conceptnet_embeddings() -> dict:
    with open(conceptnet_path, "rb") as f:
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
    if word in embeddings and (" " not in word and "," not in word):
        return np.asarray(embeddings[word], dtype=np.float32)

    # --- Compound word handling ---
    if "_" in word or "-" in word or "," in word or " " in word or "’" in word:
        if("," in word):
            parts = word.split(",")
        elif(" " in word):
            parts = word.split(" ")
        elif("’" in word):
            parts = word.split("’")
            if("’s") in word:
                parts[-1] = "’" + parts[-1]
        else:
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
                    arr = np.stack(stem_related, axis= 0).astype(np.float32)
                    part_embeddings.append(arr.mean(axis= 0))

        if part_embeddings:
            return np.mean(np.stack(part_embeddings, axis= 0), axis= 0)

    # --- Single-word stem fallback ---
    stem = stemmer.stem(word)
    stem_related = [
        v for k, v in embeddings.items()
        if stemmer.stem(k) == stem
    ]
    if stem_related:
        arr = np.stack(stem_related, axis= 0).astype(np.float32)
        return arr.mean(axis= 0)

    # --- Fallback ---
    print("Failed word: " + str(word) + "\n")
    return None

def extract_response(response: str) -> str:
    # 2. Primary: boxed answers
    boxed_pattern = re.compile(
        r'(?:/boxed\[|/boxed<|/boxed{|\\boxed\[|\\boxed<|\\boxed{)\s*(.*?)[\]>}]',
        re.DOTALL
    )
    boxed_matches = boxed_pattern.findall(response)
    if boxed_matches:
        return boxed_matches[-1]

    # 3. Secondary: explicit answer markers
    marker_pattern = re.compile(
        r'(?:/boxed\[|/boxed<|/boxed{|\\boxed\[|\\boxed<|\\boxed{)\s*(.*?)[\]>}]',
        re.IGNORECASE
    )
    marker_match = marker_pattern.search(response)
    if marker_match:
        return marker_match.group(1)
    
     # 1. Strip reasoning section if present
    if "</think>" in response:
        response = response.split("</think>", 1)[1]
        boxed_matches = boxed_pattern.findall(response)
        if boxed_matches:
            return boxed_matches[-1]

        marker_match = marker_pattern.search(response)
        if marker_match:
            return marker_match.group(1)
        
    #Default
    return "E"

def get_current_text(question: str, is_file: bool) -> str:
    q = question.lower()

    #robustly extract between TEXT and QUESTION
    if(is_file):
        match = re.search(r"TEXT:\s*(.*?)\s*QUESTION", question, re.DOTALL)
        if not match:
            return ""

        text = match.group(1)
    else:
        text = question
    #normalize whitespace
    text = re.sub(r"\s+", " ", text)

    #remove punctuation except letters and numbers
    text = re.sub(r"[^a-z0-9 ]", "", text)

    return text.strip()

def exact_match_bool(golden: str, response: str) -> bool:
    return (golden.lower() == response.lower())

def substring_match_bool(golden: str, response: str) -> bool:
    golden = golden.lower()
    response = response.lower()
    return ((golden in response) or (response in golden))

def semantic_similarity_score(golden: str, response: str, embeddings: dict) -> float:
    if(response == "E"):
        print("couldn't parse answer...\n")
        return float('NaN')
    else:
        response_embedding = get_features(response, embeddings)
        golden_embedding = get_features(golden, embeddings)
        if(response_embedding is None or golden_embedding is None):
            print("Failed to get an embedding...\n")
            return float('NaN')
        else:
            return cos_sim(response_embedding, golden_embedding)

if __name__ == "__main__":
    embeddings = get_conceptnet_embeddings()
    for file in files:
        old = pd.read_csv(home + "metaphor_project/annotations/for_question_generation.csv")
        curr = pd.read_csv(PATH + file)
        curr["cleaned_text"] = curr["mcq_prompt"].apply(lambda x: get_current_text(x, True))
        old["cleaned_text"] = old["current_text"].apply(lambda x: get_current_text(x, False))

        merged = curr.merge(old, left_on= "cleaned_text", right_on= "cleaned_text")
        merged = merged[["open_prompt", "full_answer", "current_text", "target", "original_target"]]
        merged["specific_target"] = merged["target"]
        merged["dataset_target"] = merged["original_target"]
        merged.drop(columns= ["target", "original_target"])

        merged["response"] = merged["full_answer"].apply(lambda x: extract_response(x))

        merged["dataset_match"] = merged.apply(lambda x: exact_match_bool(x["dataset_target"], x["response"]), axis= 1)
        merged["specific_match"] = merged.apply(lambda x: exact_match_bool(x["specific_target"], x["response"]), axis= 1)

        merged["dataset_substr_match"] = merged.apply(lambda x: substring_match_bool(x["dataset_target"], x["response"]), axis= 1)
        merged["specific_substr_match"] = merged.apply(lambda x: substring_match_bool(x["specific_target"], x["response"]), axis= 1)

        merged["dataset_sim"] = merged.apply(lambda x: semantic_similarity_score(x["dataset_target"], x["response"], embeddings), axis= 1)
        merged["specific_sim"] = merged.apply(lambda x: semantic_similarity_score(x["specific_target"], x["response"], embeddings), axis= 1)

        file_title = file[:file.index("-target")]

        merged.to_csv(f"detection_results/{file_title}.csv", index= False)

        print(f"Results saved to {file_title}.csv!\n")
