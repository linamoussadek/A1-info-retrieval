import json
import pytrec_eval

def load_relevance(file_path):
    """Load ground truth relevance judgments (test.tsv)."""
    relevance = {}
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            query_id, _, doc_id, relevance_score = parts
            if query_id not in relevance:
                relevance[query_id] = {}
            relevance[query_id][doc_id] = int(relevance_score)
    return relevance

def load_results(file_path):
    """Load system retrieval results from Results.txt."""
    results = {}
    with open(file_path, "r") as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split()
            query_id, _, doc_id, _, score, _ = parts
            if query_id not in results:
                results[query_id] = {}
            results[query_id][doc_id] = float(score)
    return results

relevance_file = "../test.tsv"
results_file = "Results.txt"

relevance = load_relevance(relevance_file)
results = load_results(results_file)

metrics = {'map', 'P_10', 'recall_100', 'recall_20', 'ndcg'}
evaluator = pytrec_eval.RelevanceEvaluator(relevance, metrics)
evaluation_results = evaluator.evaluate(results)

average_scores = {metric: 0 for metric in metrics}
num_queries = len(evaluation_results)

for query_id in evaluation_results:
    for metric in metrics:
        average_scores[metric] += evaluation_results[query_id].get(metric, 0)

if num_queries > 0:
    for metric in average_scores:
        average_scores[metric] /= num_queries

print("Evaluation Results:")
for metric, score in average_scores.items():
    print(f"{metric}: {score:.4f}")

with open("evaluation_results.txt", "w") as f:
    for metric, score in average_scores.items():
        f.write(f"{metric}: {score:.4f}\n")

print("Evaluation results saved to evaluation_results.txt.")
