import json

def generateInvertedIndex(input_file, output_file):
    invertedIndex = {}

    # Open and load the JSON file
    with open(input_file, "r") as file:
        documents = json.load(file)
        
        # Process each document in the JSON file
        for doc in documents:
            doc_id = doc["doc_id"]
            tokens = doc["tokens"]
            
            # Create a frequency count for tokens in this document
            token_frequency = {}
            for token in tokens:
                token_frequency[token] = token_frequency.get(token, 0) + 1
            
            # Loop through the token frequency dictionary
            for token, tf in token_frequency.items():
                if token not in invertedIndex:
                    invertedIndex[token] = {}  # If token not in index, add it
                
                # For the current token, add document ID and term frequency
                invertedIndex[token][doc_id] = tf

    # Sort the inverted index lexicographically by token
    sortedInvertedIndex = dict(sorted(invertedIndex.items()))

    # Write the sorted inverted index to the output file
    with open(output_file, 'w') as f:
        json.dump(sortedInvertedIndex, f, indent=4)



if __name__ == "__main__":
    input_file = "../scifact/preprocessed_corpus.json"
    output_file = "../scifact/invertedIndex.json"
    generateInvertedIndex(input_file, output_file)

    