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



### File IO and processing

class Document(NamedTuple):
    doc_id: int
    author: List[str]
    title: List[str]
    keyword: List[str]
    abstract: List[str]

    def sections(self):
        return [self.author, self.title, self.keyword, self.abstract]

    def __repr__(self):
        return (f"doc_id: {self.doc_id}\n" +
            f"  author: {self.author}\n" +
            f"  title: {self.title}\n" +
            f"  keyword: {self.keyword}\n" +
            f"  abstract: {self.abstract}")


def read_stopwords(file):
    with open(file) as f:
        return set([x.strip() for x in f.readlines()])

stopwords = read_stopwords('common_words')

stemmer = SnowballStemmer('english')

def read_rels(file):
    '''
    Reads the file of related documents and returns a dictionary of query id -> list of related documents
    '''
    rels = defaultdict(list)
    with open(file) as f:
        for line in f:
            qid, rel = line.strip().split()
            rels[int(qid)].append(int(rel))
    return dict(rels)

def read_docs(file):
    '''
    Reads the corpus into a list of Documents
    '''
    docs = [defaultdict(list)]  # empty 0 index
    category = ''
    with open(file) as f:
        i = 0
        for line in f:
            line = line.strip()
            if line.startswith('.I'):
                i = int(line[3:])
                docs.append(defaultdict(list))
            elif re.match(r'\.\w', line):
                category = line[1]
            elif line != '':
                for word in word_tokenize(line):
                    docs[i][category].append(word.lower())

    return [Document(i + 1, d['A'], d['T'], d['K'], d['W'])
        for i, d in enumerate(docs[1:])]

def stem_doc(doc: Document):
    return Document(doc.doc_id, *[[stemmer.stem(word) for word in sec]
        for sec in doc.sections()])

def stem_docs(docs: List[Document]):
    return [stem_doc(doc) for doc in docs]

def remove_stopwords_doc(doc: Document):
    return Document(doc.doc_id, *[[word for word in sec if word not in stopwords]
        for sec in doc.sections()])

def remove_stopwords(docs: List[Document]):
    return [remove_stopwords_doc(doc) for doc in docs]



### Term-Document Matrix

class TermWeights(NamedTuple):
    author: float
    title: float
    keyword: float
    abstract: float

def compute_doc_freqs(docs: List[Document]):
    '''
    Computes document frequency, i.e. how many documents contain a specific word
    '''
    freq = Counter()
    for doc in docs:
        words = set()
        for sec in doc.sections():
            for word in sec:
                words.add(word)
        for word in words:
            freq[word] += 1
    return freq

def compute_tf(doc: Document, doc_freqs: Dict[str, int], weights: list, N: int):
    vec = defaultdict(float)
    for word in doc.author:
        vec[word] += weights.author
    for word in doc.keyword:
        vec[word] += weights.keyword
    for word in doc.title:
        vec[word] += weights.title
    for word in doc.abstract:
        vec[word] += weights.abstract
    return dict(vec)  # convert back to a regular dict

def compute_tfidf(doc, doc_freqs, weights, N):
    vec = compute_tf(doc, doc_freqs, weights, N)
    for word in vec.keys():
        if vec[word] > 0:
            vec[word] *= np.log2(N / (doc_freqs[word] + 0.000001))
    return vec

def compute_boolean(doc, doc_freqs, weights, N):
    return {word: 1 for word in set(doc.author + doc.title + doc.keyword + doc.abstract)}



### Vector Similarity
def keys(x: Dict[str, float], y: Dict[str, float]) -> set[str]:
    '''
    Returns the keys from the smaller of two dictionaries.
    '''
    return list(x.keys()) if len(x) < len(y) else list(y.keys())

def dictdot(x: Dict[str, float], y: Dict[str, float]) -> float:
    '''
    Computes the dot product of vectors x and y, represented as sparse dictionaries.
    '''
    return sum(x.get(key, 0) * y.get(key, 0) for key in keys(x, y))

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


### Precision/Recall

def interpolate(x1, y1, x2, y2, x):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m * x + b

def precision_at(recall: float, results: List[int], relevant: List[int]) -> float:
    '''
    This function should compute the precision at the specified recall level.
    If the recall level is in between two points, you should do a linear interpolation
    between the two closest points. For example, if you have 4 results
    (recall 0.25, 0.5, 0.75, and 1.0), and you need to compute recall @ 0.6, then do something like

    interpolate(0.5, prec @ 0.5, 0.75, prec @ 0.75, 0.6)

    Note that there is implicitly a point (recall=0, precision=1).

    `results` is a sorted list of document ids
    `relevant` is a list of ids of relevant documents for the query
    '''
    relevant_len = len(relevant)
    recalls = [i / relevant_len for i in range(relevant_len + 1)]
    ranks = sorted(results.index(doc_id) + 1 for doc_id in relevant)
    precisions = [(i+1) / rank for i, rank in enumerate(ranks)]
    precisions.insert(0, 1)
    
    recall_level = recall * relevant_len
    if recall_level != int(recall_level):
        lower = math.floor(recall_level) 
        upper = math.ceil(recall_level)
        return interpolate(recalls[lower], precisions[lower], recalls[upper], precisions[upper], recall)
    else:
        return precisions[int(recall_level)]

def mean_precision1(results, relevant):
    return (precision_at(0.25, results, relevant) +
        precision_at(0.5, results, relevant) +
        precision_at(0.75, results, relevant)) / 3

def mean_precision2(results, relevant):
    return sum(precision_at(i / 10, results, relevant) for i in range(1, 11)) / 10

def norm_recall(results, relevant):
    rel = len(relevant)
    N = len(results)

    relevant_positions = [results.index(doc_id) + 1 for doc_id in relevant]
    
    sum_ranks = sum(relevant_positions)
    sum_integers = sum(range(1, rel + 1))

    return 1 - ((sum_ranks - sum_integers) / (rel * (N - rel)))

def norm_precision(results, relevant):
    rel = len(relevant)
    N = len(results)
    
    relevant_positions = [results.index(doc_id) + 1 for doc_id in relevant]
    
    sum_ranks = sum(np.log2(pos) for pos in relevant_positions)
    sum_integers = sum(np.log2(i) for i in range(1, rel + 1))
    denom = N * np.log2(N) - (N - rel) * np.log2(N - rel) - rel * np.log2(rel)
    
    return 1 - (sum_ranks - sum_integers) / denom
    
    


### Extensions

# TODO: put any extensions here
def svd(doc_vectors: List[Dict[str, float]], doc_freq: Dict[str, int], N: int, M: int, proceed: bool) -> List[Dict[str, float]]:
    '''
    Perform SVD on the N terms x M documents term-document matrix and return a new list of document vectors
    '''
    if not proceed:
        return doc_vectors   
    
    matrix = np.zeros((N, M))      # N x M matrix
    terms = list(doc_freq.keys())
    term_to_index = {term: i for i, term in enumerate(terms)}
    
    # Turn list of dicts into a matrix
    for i, doc_vec in enumerate(doc_vectors):
        for term, freq in doc_vec.items():
            matrix[term_to_index[term], i] = freq
            
    # SVD 
    T, S, D = np.linalg.svd(matrix, full_matrices=False)
    
    # Find K based off 90% optimization threshold
    sum_sq = np.dot(S, S)
    K = 1
    for k in range(1, S.shape[0]):
        if np.dot(S[:k], S[:k]) / sum_sq >= 0.9:
            K = k
            break
    
    # Truncate 
    T = T[:, :K]     # N x K
    S = S[:K]        # K
    D = D[:K, :]     # K x M
    matrix = T @ np.diag(S) @ D
    
    # Reconstruct
    new_doc_vectors = []
    for i in range(M):
        new_doc_vec = {}
        for j in range(N):
            new_doc_vec[terms[j]] = matrix[j, i]
        new_doc_vectors.append(new_doc_vec)
    
    return new_doc_vectors
    
        


### Search

def experiment():
    q = open('input_query.raw', 'w')
    count = 0
    print('Enter a query or type "exit" to finish')
    while True:
        if input('Continue? ').strip() == 'exit':
            break
        
        q.write(f'.I {count + 1}\n')
        print(f'Query: {count + 1}\n')
        
        q.write('.A\n')
        print('Enter relevant author(s) of the query or type "exit" to finish')
        while True:
            if input('Continue? ').strip() == 'exit':
                break
            authors = input('Author: \n').strip()
            q.write(authors + '\n')
        
        q.write('.T\n')
        print('Enter relevant title(s) or type "exit" to finish')
        while True:
            if input('Continue? ').strip() == 'exit':
                break
            title = input('Title: \n').strip()
            q.write(title + '\n')
        
        q.write('.K\n')
        print('Enter relevant keyword(s) or type "exit" to finish')
        while True:
            if input('Continue? ').strip() == 'exit':
                break
            keyword = input('Keyword: \n').strip()
            q.write(keyword + '\n')
        
        s = input('Search: \n').strip()
        q.write('.W\n')
        q.write(s + '\n')

        q.write('\n')
        count += 1
    q.close()

    docs = read_docs('cacm.raw')
    queries = read_docs('input_query.raw')
    rels = read_rels('query.rels')
    stopwords = read_stopwords('common_words')

    term_funcs = {
        #'tf': compute_tf,
        'tfidf': compute_tfidf,
        #'boolean': compute_boolean
    }

    sim_funcs = {
        'cosine': cosine_sim,
        #'jaccard': jaccard_sim,
        #'dice': dice_sim,
        #'overlap': overlap_sim
    }

    permutations = [
        term_funcs, 
        #[False, True],  # stem
        [True],
        #[False, True],  # remove stopwords
        [True],
        sim_funcs,
        [TermWeights(author=1, title=1, keyword=1, abstract=1)],
            #TermWeights(author=1, title=3, keyword=4, abstract=1),
            #TermWeights(author=1, title=1, keyword=1, abstract=4)]
        [False, True]     # SVD
    ]

    print('term', 'stem', 'removestop', 'sim', 'termweights', 'svd', 'p_0.25', 'p_0.5', 'p_0.75', 'p_1.0', 'p_mean1', 'p_mean2', 'r_norm', 'p_norm', sep='\t')

    # This loop goes through all permutations. You might want to test with specific permutations first
    for term, stem, removestop, sim, term_weights, proceed in itertools.product(*permutations):
        processed_docs, processed_queries = process_docs_and_queries(docs, queries, stem, removestop)
        doc_freqs = compute_doc_freqs(processed_docs)
        M = len(processed_docs)   # number of documents
        doc_vectors = [term_funcs[term](doc, doc_freqs, term_weights, M) for doc in processed_docs]
        N = len(doc_freqs)        # number of unique terms
        doc_vectors = svd(doc_vectors, doc_freqs, N, M, proceed=proceed)
        
        metrics = []
        for query in processed_queries:
            query_vec = term_funcs[term](query, doc_freqs, term_weights, M)
            results = search(doc_vectors, query_vec, sim_funcs[sim])
            #results = search_debug(processed_docs, query, rels[query.doc_id], doc_vectors, query_vec, sim_funcs[sim])
            rel = rels[query.doc_id] 
             
            metrics.append([
                precision_at(0.25, results, rel),
                precision_at(0.5, results, rel),
                precision_at(0.75, results, rel),
                precision_at(1.0, results, rel),
                mean_precision1(results, rel),
                mean_precision2(results, rel),
                norm_recall(results, rel),
                norm_precision(results, rel)
            ])
        
        averages = [f'{np.mean([metric[i] for metric in metrics]):.4f}' for i in range(len(metrics[0]))]
        print(term, stem, removestop, sim, ','.join(map(str, term_weights)), proceed, *averages, sep='\t')
     
        '''
        print('STEMMING:', stem)
        writeup_queries = [processed_queries[5], processed_queries[8], processed_queries[21]]
        for query in writeup_queries:
            query_vec = term_funcs[term](query, doc_freqs, term_weights, len(processed_docs))
            rel = rels[query.doc_id] 
            results_with_score = [(doc_id + 1, sim_funcs[sim](query_vec, doc_vec)) for doc_id, doc_vec in enumerate(doc_vectors)]
            results_with_score = sorted(results_with_score, key=lambda x: -x[1])
            print('Query:', query)
            for doc_id, score in results_with_score[:20]:
                if doc_id in rel:
                    print('*****')
                print('Doc ID:', doc_id)
                print('Cosine Score:', score)
                print('Title:', processed_docs[doc_id - 1].title)
                if doc_id in rel:
                    print('*****')
                print()
            print()
        '''
        
        
        '''
        print('STEMMING:', stem)
        for query in processed_queries:
            query_vec = term_funcs[term](query, doc_freqs, term_weights, len(processed_docs))
            results_with_score = [(doc_id + 1, sim_funcs[sim](query_vec, doc_vec)) for doc_id, doc_vec in enumerate(doc_vectors)]
            results_with_score = sorted(results_with_score, key=lambda x: -x[1])
            print('Query:', query)
            for doc_id, score in results_with_score[:10]:
                print('Doc ID:', doc_id)
                print('Term-Weight Pairs:', doc_vectors[doc_id - 1])
                print()
            print()
        '''
        
        '''
        print('STEMMING:', stem)
        writeup_docs = [processed_docs[238], processed_docs[1235], processed_docs[2739]]
        for DOC in writeup_docs:
            DOC_vec = doc_vectors[DOC.doc_id - 1]
            results_with_score = [(doc_id + 1, sim_funcs[sim](DOC_vec, doc_vec)) for doc_id, doc_vec in enumerate(doc_vectors)]
            results_with_score = sorted(results_with_score, key=lambda x: -x[1])
            print('Comparison Doc:', DOC.doc_id)
            print()
            for doc_id, score in results_with_score[1:21]:
                print('Doc ID:', doc_id)
                print('Cosine Score:', score)
                print('Title:', processed_docs[doc_id - 1].title)
                print()
            print()
        '''
        

def process_docs_and_queries(docs, queries, stem, removestop):
    processed_docs = docs
    processed_queries = queries
    if removestop:
        processed_docs = remove_stopwords(processed_docs)
        processed_queries = remove_stopwords(processed_queries)
    if stem:
        processed_docs = stem_docs(processed_docs)
        processed_queries = stem_docs(processed_queries)
    return processed_docs, processed_queries


def search(doc_vectors, query_vec, sim):
    results_with_score = [(doc_id + 1, sim(query_vec, doc_vec))
                    for doc_id, doc_vec in enumerate(doc_vectors)]
    results_with_score = sorted(results_with_score, key=lambda x: -x[1])
    results = [x[0] for x in results_with_score]
    return results


def search_debug(docs, query, relevant, doc_vectors, query_vec, sim):
    results_with_score = [(doc_id + 1, sim(query_vec, doc_vec))
                    for doc_id, doc_vec in enumerate(doc_vectors)]
    results_with_score = sorted(results_with_score, key=lambda x: -x[1])
    results = [x[0] for x in results_with_score]

    print('Query:', query)
    print('Relevant docs: ', relevant)
    print()
    for doc_id, score in results_with_score[:10]:
        print('Score:', score)
        print(docs[doc_id - 1])
        print()
    return results


if __name__ == '__main__':
    experiment()