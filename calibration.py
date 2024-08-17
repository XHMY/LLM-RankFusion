import numpy as np
from os.path import join
from glob import glob
from tqdm import tqdm
from scipy.special import softmax
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="outputs/run.pairwise*preference_matrix")
args = parser.parse_args()

token_prob_df = []
model_preference_mat = {}
ori_model_preference_mat = {}
for model_dir in tqdm(glob(args.model_dir)):
    mean_list = []
    preference_mat = {}
    ori_preference_mat = {}
    for q_file in glob(join(model_dir, "q-*_logit.npy")):
        qid = re.search(r"q-(\d+)_logit\.npy", q_file).group(1)
        logit_arr = np.load(q_file).astype(np.float32)
        preference_mat[qid] = softmax(logit_arr, -1)[...,0]
        preference_mat[qid] = softmax(np.stack([preference_mat[qid],
                                                preference_mat[qid].T], -1), -1)[...,0]
        np.save(q_file.replace("_logit", "_fix"), preference_mat[qid]>0.5)
        np.save(q_file.replace("_logit", "_calogit"), preference_mat[qid])
        mean_list.append(logit_arr.mean((0,1)))
        # os.rename(q_file.replace("_logit", ""), q_file.replace("_logit", "_ori")) # change old file name
        ori_preference_mat[qid] = np.load(q_file.replace("_logit", "_ori"))
    model_name = re.search(r"allpair\.(.+)_preference_matrix", model_dir).group(1)
    ori_model_preference_mat[model_name + ("-icl" if "icl" in model_dir else "")] = ori_preference_mat
    model_preference_mat[model_name + ("-icl" if "icl" in model_dir else "")] = preference_mat
    prob_value = np.stack(mean_list).mean(0)
    token_prob_df.append({
        "model": model_name, "ICL": "icl" in model_dir,
        "A": prob_value[0], "B": prob_value[1],
        "#Query": len(mean_list)
    })

print("Calibration Finished")