import json
import pytrec_eval

# Load ground truth relevance file (test.tsv)
def load_relevance(file_path):
    relevance = {}
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")  # Ensure TSV format
            if len(parts) >= 4:  # Ignore second column
                query_id, _, doc_id, relevance_score = parts  # Skip second column
                if query_id not in relevance:
                    relevance[query_id] = {}
                relevance[query_id][doc_id] = int(relevance_score)
    return relevance

# Load system results file (Results.txt)
def load_results(file_path):
    results = {}
    with open(file_path, "r") as f:
        next(f)  # Skip the header line
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                query_id, _, doc_id, rank, score, *rest = parts  # Flexible handling
                if query_id not in results:
                    results[query_id] = {}
                results[query_id][doc_id] = float(score)  # Convert score to float
    return results

# File paths (updated to correct locations)
relevance_file = "../test.tsv"   # Ground truth file (test.tsv)
results_file = "../Results.txt"  # Your system's retrieved results

# Load data
relevance = load_relevance(relevance_file)
results = load_results(results_file)

# Define evaluation metrics (Fix recall issue)
metrics = {'map', 'P_10', 'recall_100'}  # Use recall_100 instead of recall
evaluator = pytrec_eval.RelevanceEvaluator(relevance, metrics)

# Run evaluation
evaluation_results = evaluator.evaluate(results)

# Check for zero queries and prevent division by zero
num_queries = len(evaluation_results)
if num_queries == 0:
    print("Error: No queries found in evaluation. Check Results.txt and test.tsv.")
    exit(1)

# Compute and print average scores
average_scores = {metric: 0 for metric in metrics}
for query_id in evaluation_results:
    for metric in metrics:
        if metric in evaluation_results[query_id]:  # Ensure metric exists
            average_scores[metric] += evaluation_results[query_id][metric]

for metric in average_scores:
    average_scores[metric] /= num_queries  # Compute average across all queries

# Print final results
print("\nEvaluation Results:")
for metric, score in average_scores.items():
    print(f"{metric}: {score:.4f}")

# Save results to a file
with open("evaluation_results.txt", "w") as f:
    for metric, score in average_scores.items():
        f.write(f"{metric}: {score:.4f}\n")

print("\nEvaluation results saved to evaluation_results.txt")
