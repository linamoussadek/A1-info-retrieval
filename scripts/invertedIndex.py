import json
import math

def generateInvertedIndex(input_file, output_file):
    invertedIndex = {}
    doc_lengths = {}
    doc_count = 0  # For IDF calculation

    with open(input_file, "r") as file:
        documents = json.load(file)
        doc_count = len(documents)

        for doc in documents:
            doc_id = doc["doc_id"]
            tokens = doc["tokens"]
            doc_lengths[doc_id] = len(tokens)

            token_frequency = {}
            for token in tokens:
                token_frequency[token] = token_frequency.get(token, 0) + 1

            for token, tf in token_frequency.items():
                if token not in invertedIndex:
                    invertedIndex[token] = {}

                invertedIndex[token][doc_id] = tf

    # Compute IDF values for TF-IDF weighting
    idf_values = {term: math.log(doc_count / len(postings)) for term, postings in invertedIndex.items()}

    with open(output_file, 'w') as f:
        json.dump({"index": invertedIndex, "doc_lengths": doc_lengths, "idf": idf_values}, f, indent=4)

if __name__ == "__main__":
    input_file = "../scifact/preprocessed_corpus.json"
    output_file = "../scifact/invertedIndex.json"

    generateInvertedIndex(input_file, output_file)
    print("Inverted index created successfully.")
