import json
import math
from collections import defaultdict


def compute_tf_idf(inverted_index, num_docs):
    """ Compute TF-IDF weights for each word-document pair """
    tf_idf = defaultdict(dict)
    
    for term, doc_dict in inverted_index.items():
        df = len(doc_dict)  # Document frequency
        idf = math.log((num_docs / (df+1)),2)  # IDF with smoothing
        
        for doc_id, positions in doc_dict.items():
            tf = positions  # Term frequency
            tf_idf[doc_id][term] = (1 + math.log(tf)) * idf  # TF-IDF calculation
    
    return tf_idf


#code based upon the lectures and this website https://blog.christianperone.com/2011/09/machine-learning-text-feature-extraction-tf-idf-part-i/
def compute_cosine_similarity(query_tokens, inverted_index, tf_idf):
    """ Compute cosine similarity between query and documents """
    query_tf = defaultdict(int)
    for term in query_tokens:
        query_tf[term] += 1
    
    query_norm = math.sqrt(sum((1 + math.log(tf))**2 for tf in query_tf.values()))
    
    doc_scores = defaultdict(float)
    for term in query_tokens:
        if term in inverted_index:
            for doc_id in inverted_index[term]:
                doc_scores[doc_id] += (1 + math.log(query_tf[term])) * tf_idf[doc_id].get(term, 0)
    
    for doc_id in doc_scores:
        doc_norm = math.sqrt(sum(val**2 for val in tf_idf[doc_id].values()))
        if doc_norm > 0:
            doc_scores[doc_id] /= (query_norm * doc_norm)
    
    return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

def retrieve_and_rank(query, inverted_index, num_docs):
    """ Process query and return ranked documents """
    query_tokens = query.split() 
    tf_idf = compute_tf_idf(inverted_index, num_docs)
    ranked_docs = compute_cosine_similarity(query_tokens, inverted_index, tf_idf)
    return ranked_docs

def load_inverted_index(index_file):
    with open(index_file, 'r') as f:
        return json.load(f)
    
def load_processed_corpus(processed_file):
    with open(processed_file, 'r') as f:
        return json.load(f)

def load_queries(query_file):
    with open(query_file, 'r') as f:
        return [json.loads(line) for line in f]  # Load first 20 queries

def main():
    index_file = "../scifact/invertedIndex.json"
    query_file = "../scifact/queries.jsonl"
    count_file = "../scifact/processed_corpus.json"
    output_file = "Results.txt"
    
    inverted_index = load_inverted_index(index_file)
    queries = load_queries(query_file)
    countFile = load_processed_corpus(count_file)
    num_docs = 0
    for document in countFile:
        num_docs+=1
    
    with open(output_file, "w") as f:
        f.write(f"Query ID | Q0 | doc ID | ranking | score | Tag\n")
        for query in queries:
            query_id = query["_id"]
            query_text = query["text"]
            ranked_results = retrieve_and_rank(query_text, inverted_index, num_docs)
            for rank, (doc_id, score) in enumerate(ranked_results[:100], start=1):
                f.write(f"{query_id} Q0 {doc_id} {rank} {score:.4f} Version 1.2\n")
    
if __name__ == "__main__":
    main()
