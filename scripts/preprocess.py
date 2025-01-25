import json
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def tokenize(text):
    return re.findall(r'\b\w+\b', text)

def normalize_case(tokens):
    """converts tokens to lowercase."""
    return [token.lower() for token in tokens]

def remove_stopwords(tokens, stopwords_set):
    return [token for token in tokens if token not in stopwords_set]

def stem_tokens(tokens):
    """applies stemming to tokens using Porter Stemmer."""
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

def preprocess_text(text, stopwords_set):
    tokens = tokenize(text)
    tokens = normalize_case(tokens)
    tokens = remove_stopwords(tokens, stopwords_set)
    tokens = stem_tokens(tokens)
    return tokens

def preprocess_corpus(corpus_file, stopwords_file, output_file):
    with open(stopwords_file, 'r') as f:
        stopwords_set = set(f.read().splitlines())

    preprocessed_data = []
    with open(corpus_file, 'r') as f:
        for line in f:
            doc = json.loads(line.strip())
            combined_text = doc.get("title", "") + " " + doc.get("text", "")
            tokens = preprocess_text(combined_text, stopwords_set)
            preprocessed_data.append({"doc_id": doc["_id"], "tokens": tokens})

    with open(output_file, 'w') as f:
        json.dump(preprocessed_data, f)

if __name__ == "__main__":
    corpus_file = "../scifact/corpus.jsonl"
    stopwords_file = "../stopwords.txt"
    output_file = "../scifact/preprocessed_corpus.json"

    preprocess_corpus(corpus_file, stopwords_file, output_file)
    print("Preprocessing complete. Output saved to", output_file)
