import os
import json

def generateInvertedIndex(input_file, output_file):
    invertedIndex={}

    # Open and load the JSON file
    with open(input_file, "r") as file:
        documents = json.load(file)
        
        # Process each document in the JSON file
        for doc in documents:
            doc_id = doc["doc_id"]
            tokens = doc["tokens"]
            
            # Loop through tokens in the document
            for position, token in enumerate(tokens):
                if token not in invertedIndex:
                    invertedIndex[token] = {}  # If token not in index, add it
                
                # For the current token, add document ID and position
                if doc_id not in invertedIndex[token]:
                    invertedIndex[token][doc_id] = []
                
                invertedIndex[token][doc_id].append(position)

    sortedInvertedIndex = dict(sorted(invertedIndex.items()))

    with open(output_file, 'w') as f:
        json.dump(sortedInvertedIndex, f)


if __name__ == "__main__":
    input_file = "../scifact/preprocessed_corpus.json"
    output_file = "../scifact/invertedIndex.json"
    generateInvertedIndex(input_file, output_file)

    