import itertools
import re
from collections import Counter, defaultdict
from typing import Dict, List, NamedTuple

import math
import numpy as np
from numpy.linalg import norm
import nltk
nltk.download('punkt')
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize


### Positional Weighting
def uniform_weights(text: List[str], token_pos: int) -> Dict[str, float]:
    vec = {word: 1 for word in text}
    vec[text[token_pos]] = 0
    return vec

def expndecay_weights(text: List[str], token_pos: int) -> Dict[str, float]:
    vec = {}
    vec[text[token_pos]] = 0
    for i, word in enumerate(text[token_pos + 1:]): 
        vec[word] = 1 / (i + 1)
    for i, word in enumerate(text[:token_pos][::-1]):
        vec[word] = 1 / (i + 1)    
    return vec

def stepped_weights(text: List[str], token_pos: int) -> Dict[str, float]:
    vec = {}
    vec[text[token_pos]] = 0
    for i, word in enumerate(text[token_pos + 1:]): 
        if i == 0:
            vec[word] = 6
        elif i == 1 or i == 2:
            vec[word] = 3
        else:
            vec[word] = 1
    for i, word in enumerate(text[:token_pos][::-1]):
        if i == 0:
            vec[word] = 6
        elif i == 1 or i == 2:
            vec[word] = 3
        else:
            vec[word] = 1
    return vec

def logdecay_weights(text: List[str], token_pos: int) -> Dict[str, float]:
    vec = {}
    vec[text[token_pos]] = 0
    for i, word in enumerate(text[token_pos + 1:]):
        vec[word] = 1 / (math.log2(i + 2))
    for i, word in enumerate(text[:token_pos][::-1]):
        vec[word] = 1 / (math.log2(i + 2))
    return vec


### File IO and processing

class Document(NamedTuple):
    doc_id: int
    label: int
    text: List[str]
    token_pos: int

stemmer = SnowballStemmer('english')

def find_token_pos(text: List[str]) -> int:
    for i, word in enumerate(text):
        match = re.search(r'\.X-\w+', word)
        if match:
            return i
    return -1

def read_docs(file, adj):
    '''
    Reads the corpus into a list of Documents
    '''
    docs = [] 
    with open(file) as f:
        for line in f:
            text = word_tokenize(line)
            doc_id = int(text.pop(0))
            label = int(text.pop(0))
            token_pos = find_token_pos(text)
            if adj:
                if token_pos > 0:
                    text[token_pos - 1] = 'L-' + text[token_pos - 1]  # Left Adjacent
                if token_pos + 1 < len(text):
                    text[token_pos + 1] = 'R-' + text[token_pos + 1]  # Right Adjacent
            docs.append(Document(doc_id, label, text, token_pos))
    return docs

def stem_doc(doc: Document):
    return Document(doc.doc_id, doc.label, [stemmer.stem(word) for word in doc.text], doc.token_pos)

def stem_docs(docs: List[Document]):
    return [stem_doc(doc) for doc in docs]


### Term-Document Matrix

def compute_doc_freqs(docs: List[Document]):
    '''
    Computes document frequency, i.e. how many documents contain a specific word
    '''
    freq = Counter()
    for doc in docs:
        words = set()
        for word in doc.text:
            words.add(word)
        for word in words:
            freq[word] += 1
    return freq

def compute_tf(doc: Document, weight_func):
    '''
    Computes the term frequency of a document
    '''     
    if doc.token_pos != -1:
        return weight_func(doc.text, doc.token_pos)
    else:
        return {word: 1 for word in doc.text}

def compute_tfidf(doc, doc_freqs, weight_func, N):
    vec = compute_tf(doc, weight_func)
    for word in vec.keys():
        if vec[word] > 0:
            vec[word] *= math.log2(N / (doc_freqs[word] + 0.000001))
    return vec


### Vector Similarity

def dictdot(x: Dict[str, float], y: Dict[str, float]) -> float:
    '''
    Computes the dot product of vectors x and y, represented as sparse dictionaries.
    '''
    keys = list(x.keys()) if len(x) < len(y) else list(y.keys())
    return sum(x.get(key, 0) * y.get(key, 0) for key in keys)

def cosine_sim(x, y) -> float:
    '''
    Computes the cosine similarity between two sparse term vectors represented as dictionaries.
    '''
    num = dictdot(x, y)
    if num == 0:
        return 0
    return num / (norm(list(x.values())) * norm(list(y.values())))

def dice_sim(x, y) -> float:
    '''
    Computes the Dice similarity between two sparse term vectors represented as dictionaries.
    '''
    num = 2 * dictdot(x, y)
    if num == 0:
        return 0
    return num / (sum(x.values()) + sum(y.values()))

def jaccard_sim(x, y) -> float:
    '''
    Computes the Jaccard similarity between two sparse term vectors represented as dictionaries.
    '''
    num = dictdot(x, y)
    if num == 0:
        return 0
    return num / (sum(x.values()) + sum(y.values()) - num + 0.000001)

def overlap_sim(x, y) -> float:
    '''
    Computes the overlap similarity between two sparse term vectors represented as dictionaries.
    '''
    num = dictdot(x, y)
    if num == 0:
        return 0
    return num / min(sum(x.values()), sum(y.values()))



### Search

def experiment():
    files = ['tank', 'plant', 'perplace', 'smsspam']      

    permutations = [
        # 1-6
        [False, uniform_weights, False, cosine_sim],
        [True, expndecay_weights, False, cosine_sim],
        [False, expndecay_weights, False, cosine_sim],
        [False, expndecay_weights, True, cosine_sim],
        [False, stepped_weights, False, cosine_sim],
        [False, logdecay_weights, False, cosine_sim],    
        # 7-12
        [False, uniform_weights, False, jaccard_sim],
        [True, expndecay_weights, False, jaccard_sim],
        [False, expndecay_weights, False, dice_sim],
        [False, expndecay_weights, True, dice_sim],
        [False, stepped_weights, False, overlap_sim],
        [False, logdecay_weights, False, overlap_sim],   
    ]

    print('Run', 'Stemming', "Position Weighting", "Local Collocation Model", 'Similarity', "Tank", "Plant", "Perplace", "Smsspam", sep='\t')

    for run, (stem, weight_func, adj, sim) in enumerate(permutations):
        metrics = []
        for file in files:
            # Processing
            train_docs = read_docs(f'{file}-train.tsv', adj)
            dev_docs = read_docs(f'{file}-dev.tsv', adj)
            processed_train, processed_dev = process_train_dev(train_docs, dev_docs, stem)
            
            # Training
            doc_freqs = compute_doc_freqs(processed_train)
            N = len(doc_freqs)
            doc_vec1 = []
            doc_vec2 = []
            for doc in processed_train:
                tfidf = compute_tfidf(doc, doc_freqs, weight_func, N)
                if doc.label == 1:
                    doc_vec1.append(tfidf)
                else:
                    doc_vec2.append(tfidf)
            
            profile1 = {}
            profile2 = {}
            for word in doc_freqs.keys():
                words1 = [doc[word] for doc in doc_vec1 if word in doc]
                if words1:
                    profile1[word] = np.mean(words1)
                    
                words2 = [doc[word] for doc in doc_vec2 if word in doc]
                if words2:
                    profile2[word] = np.mean(words2)
            
            # Testing
            correct = 0
            for doc in processed_dev:
                dev_vec = compute_tfidf(doc, doc_freqs, weight_func, N)
                sim1 = sim(dev_vec, profile1)
                sim2 = sim(dev_vec, profile2)
                if sim1 >= sim2:
                    if doc.label == 1:
                        correct += 1
                else:
                    if doc.label == 2:
                        correct += 1
            metrics.append(correct / len(processed_dev))
        
        print(run + 1, stem, weight_func.__name__, 'adj-sep-LR' if adj else 'bag-of-words', sim.__name__, *[round(metric, 4) for metric in metrics], sep='\t')

def process_train_dev(train, dev, stem):
    processed_train = train
    processed_dev = dev
    if stem:
        processed_train = stem_docs(processed_train)
        processed_dev = stem_docs(processed_dev)
    return processed_train, processed_dev



if __name__ == '__main__':
    experiment()