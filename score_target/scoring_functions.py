'''
Calculate specificity scores for targets using metrics from 
On abstraction: decoupling conceptual concreteness and categorical specificity
Marianna Bolognesi1 · Christian Burgers2,3 · Tommaso Caselli4 
2020
'''
from nltk.corpus import wordnet as wn
import pickle
import math

try:
    with open('N.pkl', 'rb') as f:
        N = pickle.load(f)
except:
    N = sum(1 for _ in wn.all_synsets('n')) #total number of noun synsets
    with open('N.pkl', 'wb') as f:
        N = pickle.dump(N, f)

def get_depth(synset) -> int:
    # Source - https://stackoverflow.com/a
    # Posted by Ivan Gordeli
    # Retrieved 2025-12-22, License - CC BY-SA 4.0
    return synset.min_depth()

# Source - https://stackoverflow.com/a
# Posted by Stefan D
# Retrieved 2025-12-22, License - CC BY-SA 3.0
def get_hyponyms(synset) -> set:
    hyponyms = set()
    for hyponym in synset.hyponyms():
        hyponyms |= set(get_hyponyms(hyponym)) #union operation and recursion
    return hyponyms | set(synset.hyponyms()) #get total size of direct and indirect hyponyms

def get_number_of_hyponyms(synset) -> int:
    hyponyms = get_hyponyms(synset)
    return len(hyponyms)

def get_synset(word: str):
    synsets = wn.synsets(word)
    for synset in synsets:
        if('.n' in synset.name()):
            return synset
    print(f"No noun synsets found for {word}...returning empty string to skip\n")
    return ""

#d + log((1 + n)/N)
def get_first_score(word: str) -> float:
    noun_synset = get_synset(word)
    if(isinstance(noun_synset, str) == True):
        return float('NaN')
    d = get_depth(noun_synset)
    n = get_number_of_hyponyms(noun_synset) 
    arg = (1 + n) / N
    return d + math.log(arg)

#(1 + d)/20
def get_third_score(word: str) -> float:
    noun_synset = get_synset(word)
    if(isinstance(noun_synset, str) == True):
        return float('NaN')
    d = get_depth(noun_synset)
    return float((1 + d)/20)

def get_actual_third_score(word: str) -> float:
    noun_synset = get_synset(word)
    if(isinstance(noun_synset, str) == True):
        return float('NaN')
    d = get_number_of_hyponyms(noun_synset)
    return float((1 + d)/20)

