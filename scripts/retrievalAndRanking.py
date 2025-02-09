import json
import math
from collections import defaultdict

# BM25 parameters (optimized for balanced performance)
k1 = 1.5  # Scaling factor for term frequency
b = 0.75  # Normalization parameter for document length

def compute_bm25_scores(query_tokens, inverted_index, doc_lengths, avg_doc_length):
    """Compute BM25 scores for each document given a query"""
    scores = defaultdict(float)
    num_docs = len(doc_lengths)

    for term in query_tokens:
        if term in inverted_index:
            df = len(inverted_index[term])  # Document frequency
            idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1)  # Smoothed IDF

            for doc_id, tf in inverted_index[term].items():
                doc_length = doc_lengths[doc_id]
                bm25_score = idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avg_doc_length))))
                scores[doc_id] += bm25_score  # Accumulate scores

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def retrieve_and_rank(query, inverted_index, doc_lengths):
    """Process query and return BM25-ranked documents"""
    query_tokens = query.split()
    avg_doc_length = sum(doc_lengths.values()) / len(doc_lengths)  # Compute average document length
    ranked_docs = compute_bm25_scores(query_tokens, inverted_index, doc_lengths, avg_doc_length)
    return ranked_docs

def load_inverted_index(index_file):
    with open(index_file, 'r') as f:
        data = json.load(f)
    return data["index"], data["doc_lengths"]  # Load both index & document lengths

def load_queries(query_file):
    with open(query_file, 'r') as f:
        return [json.loads(line) for line in f]

def main():
    index_file = "../scifact/invertedIndex.json"
    query_file = "../scifact/queries.jsonl"
    output_file = "Results.txt"

    inverted_index, doc_lengths = load_inverted_index(index_file)
    queries = load_queries(query_file)

    with open(output_file, "w") as f:
        f.write("Query ID | Q0 | doc ID | ranking | score | Tag\n")
        for query in queries:
            query_id = query["_id"]
            query_text = query["text"]
            ranked_results = retrieve_and_rank(query_text, inverted_index, doc_lengths)

            for rank, (doc_id, score) in enumerate(ranked_results[:100], start=1):
                f.write(f"{query_id} Q0 {doc_id} {rank} {score:.4f} BM25\n")

if __name__ == "__main__":
    main()
