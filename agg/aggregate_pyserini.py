import numpy as np
import pandas as pd
import argparse
from glob import glob
from os.path import join
import json
from collections import defaultdict
from joblib import Parallel, delayed
from tqdm import tqdm
import RankAggregationLib as ralib


def read_pyserini_ranking(file_path):
    """
    Read ranking file in Pyserini format
    Returns: List of tuples (query_id, DataFrame with columns [doc_id, rank, score])
    """
    rankings = defaultdict(list)
    with open(file_path) as f:
        for line in f:
            query_id, _, doc_id, rank, score, _ = line.strip().split()
            rankings[query_id].append({
                'doc_id': doc_id,
                'rank': int(rank),
                'score': float(score)
            })

    return [(qid, pd.DataFrame(ranks)) for qid, ranks in rankings.items()]


def write_pyserini_ranking(aggregated_rankings, output_path, run_id):
    """
    Write aggregated rankings in Pyserini format
    Rankings should be sorted by rank values (ascending, as lower rank is better)
    """
    with open(output_path, 'w') as f:
        for query_id, rankings in aggregated_rankings.items():
            # Sort by rank values
            sorted_docs = sorted(rankings.items(), key=lambda x: x[1])
            for rank, (doc_id, original_rank) in enumerate(sorted_docs, 1):
                score = - rank  # Convert rank to score
                f.write(f"{query_id} Q0 {doc_id} {rank} {score} {run_id}\n")


def aggregate_query_rankings(query_id, rankings, agg_func):
    """
    Aggregate rankings for a single query using Borda Count
    Returns a tuple of (query_id, Series) where Series has document IDs as index
    and aggregated ranks as values
    """
    # Convert rankings to numpy array
    voters = rankings.columns
    items = rankings.index
    num_voters = len(voters)
    num_items = len(items)

    input_array = np.zeros((num_voters, num_items))
    for i, voter in enumerate(voters):
        input_array[i] = rankings[voter].values

    # Apply Borda Count aggregation
    aggregated_ranks = agg_func(input_array)

    # Create series with document IDs as index and ranks as values
    # Note: BordaAgg already returns ranks where lower number is better
    result = pd.Series(aggregated_ranks, index=items)
    return query_id, result


def main():
    parser = argparse.ArgumentParser(description='Aggregate rankings using Borda Count')
    parser.add_argument('--docid_map_path', type=str, required=True,
                        help='Path to directory containing docid mapping files')
    parser.add_argument('--input', type=str, required=True,
                        help='Glob pattern for input ranking files')
    parser.add_argument('--output', type=str, required=True,
                        help='Output file path for aggregated rankings')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of parallel jobs (-1 for all cores)')
    args = parser.parse_args()

    # Read all ranking files
    print("Reading ranking files...")
    ranking_files = glob(args.input)
    all_rankings = [read_pyserini_ranking(file) for file in ranking_files]

    # Read document ID mappings
    print("Reading document ID mappings...")
    docid_map_query = {}
    for file_path in glob(join(args.docid_map_path, "docid-map_q-*.json")):
        query_id = file_path.split("q-")[-1].split(".")[0]
        docid_map_query[query_id] = json.load(open(file_path))

    # Organize rankings by query
    print("Organizing rankings by query...")
    query_rankings = {}
    for file_idx, rankings in enumerate(all_rankings):
        for query_id, ranking in rankings:
            # Rearrange the rankings to be indexed by docid
            ranking = ranking.set_index('doc_id').loc[
                [i for i in docid_map_query[query_id].keys()]
            ].reset_index()

            if query_id not in query_rankings:
                query_rankings[query_id] = pd.DataFrame()

            # Add the rankings from this file as a new column
            query_rankings[query_id][f'ranker_{file_idx}'] = ranking.set_index('doc_id')['rank']


    for agg_name in tqdm(ralib.__all__):
        agg_func = getattr(ralib, agg_name)
        output_file = args.output.replace("agg.txt", f"{agg_name}.txt")

        # Aggregate rankings for each query in parallel
        print("Aggregating rankings in parallel...")
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(aggregate_query_rankings)(query_id, rankings, agg_func)
            for query_id, rankings in tqdm(query_rankings.items())
        )

        # Organize results
        aggregated_rankings = {query_id: rankings for query_id, rankings in results}

        # Write results
        print("Writing aggregated rankings...")
        write_pyserini_ranking(aggregated_rankings, output_file, 'BORDA_RUN')

        print(f"Aggregation complete. Results written to '{output_file}'.")


if __name__ == "__main__":
    main()