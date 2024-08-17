import argparse
import json
from os.path import join

import ranky as rk
import pandas as pd
from glob import glob
from collections import defaultdict


def read_pyserini_ranking(file_path):
    df = pd.read_csv(file_path, sep='\s+', header=None,
                     names=['query_id', 'Q0', 'doc_id', 'rank', 'score', 'run_id'])
    return df.groupby('query_id')


def write_pyserini_ranking(rankings, output_file, run_id):
    with open(output_file, 'w') as f:
        for query_id, ranking in rankings.items():
            for rank, (doc_id, score) in enumerate(ranking.items(), 1):
                f.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_id}\n")


def main():
    parser = argparse.ArgumentParser(description='Aggregate rankings using Borda Count')
    parser.add_argument('--docid_map_path', type=str, default='outputs/dl20/docid-map')
    parser.add_argument('--input', type=str, default='outputs/t0/run.*.txt')
    parser.add_argument('--output', type=str, default='aggregated_ranking.txt')
    args = parser.parse_args()


    # Read all ranking files
    ranking_files = glob(args.input)
    all_rankings = [read_pyserini_ranking(file) for file in ranking_files]

    docid_map_query = {} # docid to idx mapping for each query
    for file_path in glob(join(args.docid_map_path, "docid-map_q-*.json")):
        docid_map_query[int(file_path.split("-")[-1].split(".")[0])] = json.load(open(file_path))

    # Organize rankings by query
    query_rankings = defaultdict(list)
    for file_idx, rankings in enumerate(all_rankings):
        for query_id, ranking in rankings:
            # rearrange the rankings to be indexed by docid
            ranking = ranking.set_index('doc_id').loc[[int(i) for i in docid_map_query[query_id].keys()]].reset_index()
            if query_id in query_rankings.keys():
                query_rankings[query_id][file_idx] = ranking.set_index('doc_id')['rank']
            else:
                query_rankings[query_id] = pd.DataFrame({file_idx: ranking.set_index('doc_id')['rank']})

    # Aggregate rankings for each query using Borda Count
    aggregated_rankings = {}
    for query_id, rankings in query_rankings.items():
        # Apply Borda Count
        borda_scores = rk.borda(rankings)

        # Sort the results
        sorted_scores = pd.Series(borda_scores).sort_values(ascending=False)

        aggregated_rankings[query_id] = sorted_scores

    # Write the aggregated rankings in Pyserini format
    write_pyserini_ranking(aggregated_rankings, args.output, 'BORDA_RUN')

    print(f"Aggregation complete. Results written to '{args.output}'.")


if __name__ == '__main__':
    main()