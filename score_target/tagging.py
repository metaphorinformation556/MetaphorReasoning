import pandas as pd
import spacy
import time
from word_forms.word_forms import get_word_forms
from pathlib import Path

nlp = spacy.load("en_core_web_sm")

home = str(Path.home())

data = pd.read_csv(f"{home}/MetaphorReasoning/annotations/for_question_generation.csv")

def tag_target(target: str) -> list[tuple[str, str]]:
    doc = nlp(target)
    return [(token.text, token.pos_) for token in doc]

def extract_noun(poses: list[tuple[str, str]]) -> str:
    nouns = [token for token, pos in poses if pos in ("NOUN", "PROPN")]

    if len(nouns) > 1:
        sentence = " ".join(token for token, _ in poses)
        print(f"Multiple nouns found: {nouns}")
        print(f"Sentence: {sentence}")
        return ""

    if len(nouns) == 1:
        return nouns[0]

    if len(nouns) == 0:
        print("No noun found")
        print(poses)
        return ""

    return nouns[0]

def posesive_to_base(word: str):
    if "'s" in word:
        word = word.replace("'s")
    return word

def chop_hyphen(word: str):
    if "-" in word:
        word = word[word.index("-") + 1:]
    return word

def nounify(word: str):
    tags = tag_target(word)
    tag = tags[0][1]
    word = tags[0][0]
    if(tag == "NOUN" or tag == "PROPN"):
        return word
    else:
        forms = get_word_forms(word)
        return list(forms['n'])[0]

def manual_override(word: str):
    if(word == "detrimental"):
        word = "detriment"
        return word
    elif(word == "worldview"):
        word = "weltanschauung"
        return word
    else:
        return word