import json

def generateInvertedIndex(input_file, output_file):
    invertedIndex = {}
    doc_lengths = {}  # Store document lengths

    with open(input_file, "r") as file:
        documents = json.load(file)
        
        for doc in documents:
            doc_id = doc["doc_id"]
            tokens = doc["tokens"]
            doc_length = len(tokens)
            doc_lengths[doc_id] = doc_length  # Store document length

            token_frequency = {}
            for token in tokens:
                token_frequency[token] = token_frequency.get(token, 0) + 1

            for token, tf in token_frequency.items():
                if token not in invertedIndex:
                    invertedIndex[token] = {}
                invertedIndex[token][doc_id] = tf  # Store term frequency

    # Write the inverted index and document lengths
    with open(output_file, 'w') as f:
        json.dump({"index": invertedIndex, "doc_lengths": doc_lengths}, f, indent=4)

if __name__ == "__main__":
    input_file = "../scifact/preprocessed_corpus.json"
    output_file = "../scifact/invertedIndex.json"

    generateInvertedIndex(input_file, output_file)
    print("Inverted index created successfully.")
