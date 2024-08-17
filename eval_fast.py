import argparse
import json
import os.path
import platform
import subprocess
from glob import glob
from tqdm import tqdm
import pandas as pd
from joblib import Parallel, delayed


# Function to evaluate ndcg and read stats, then combine results
def evaluate_log_file(log_file):
    scores = eval_ndcg(log_file)
    stats = read_stats_json(log_file)
    file_name = log_file.split('.')
    exp_name = log_file.split('/')[-2]
    model_name = ".".join(file_name[3:-1])
    model_name = model_name.replace("msmarco-v1-passage.", "")
    return {"model": model_name, "rank_type": file_name[1], "algo_type": file_name[2], "exp_name": exp_name, **scores,
            **stats}


def eval_ndcg(log_file):
    if "dl20" in log_file:
        dataset_name = "dl20-passage"
        print("Evaluating on dl20-passage dataset")
    elif "dl19" in log_file:
        dataset_name = "dl19-passage"
        print("Evaluating on dl19-passage dataset")
    else:
        print("Unknown dataset, please specify the dataset name in the log file")
        dataset_name = "dl19-passage"

    scores = {}
    for ndcg_tag in [1, 5, 10]:
        cmd = ["python", "-m", "pyserini.eval.trec_eval",
               "-c", "-l", "2", "-m", f"ndcg_cut.{ndcg_tag}", dataset_name, log_file]
        shell = platform.system() == "Windows"
        process = subprocess.Popen(cmd,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   shell=shell)
        stdout, stderr = process.communicate()
        try:
            scores[f"ndcg@{ndcg_tag}"] = float(stdout.decode('utf-8').split()[-1])
        except ValueError:
            print("Error during parsing the evaluation output, the output looks like:", stdout.decode('utf-8'))
    return scores


def read_stats_json(log_file):
    stats_path = log_file.replace('.txt', '_stats.json')
    if not os.path.exists(stats_path):
        return {}
    with open(stats_path, 'r') as f:
        raw_stat = json.load(f)
        data = raw_stat["positional_bias_stat"]

    total = sum(data.values())
    if total == 0:
        return {}
    for k, v in data.items():
        data[k] = v / total

    del raw_stat["positional_bias_stat"]
    data.update(raw_stat)

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default="outputs/run.*.txt")
    parser.add_argument("--output_file", type=str, default="outputs/ndcg_scores.csv")
    args = parser.parse_args()

    log_files = glob(args.log_file)

    # Adding a baseline result manually
    # results = [{"model": "N/A", "rank_type": "no-rank", "algo_type": "N/A", "exp_name": "baseline",
    #             **eval_ndcg('data/run.msmarco-v1-passage.bm25-default.dl20.txt')}]
    results = []

    # Parallel processing
    parallel_results = Parallel(n_jobs=-1)(delayed(evaluate_log_file)(log_file) for log_file in tqdm(log_files))
    results.extend(parallel_results)

    df = pd.DataFrame(results).sort_values(["ndcg@10"], ascending=False)
    print(df)
    df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()
