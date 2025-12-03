# LLM-RankFusion

This repository is for the paper: [https://arxiv.org/abs/2406.00231](https://arxiv.org/abs/2406.00231)

The code is adapted from [llm-rankers](https://github.com/ielab/llm-rankers)

---
## Installation

Git clone this repository, then pip install the following libraries:

```bash
torch
transformers
pyserini
ir-datasets
openai
tiktoken
accelerate
scipy
joblib
```
> You may also need to install some pyserini dependencies such as faiss. We refer to pyserini installation doc [link](https://github.com/castorini/pyserini/blob/master/docs/installation.md#development-installation)

Please ensure you have java installed on your machine and `JAVA_HOME` is set.

---
## First-stage runs
We use LLMs to re-rank top documents retrieved by a first-stage retriever. In this repo we take BM25 as the retriever.

We rely on [pyserini](https://github.com/castorini/pyserini) IR toolkit to get BM25 ranking. 

Here is an example of using pyserini command lines to generate BM25 run files on TREC DL 2019 and TREC DL 2020 datasets:

```bash
for dataset in dl19 dl20
do
    python -m pyserini.search.lucene \
        --threads 16 --batch-size 128 \
        --index msmarco-v1-passage \
        --topics ${dataset}-passage \
        --output data/run.msmarco-v1-passage.bm25-default.${dataset}.txt \
        --bm25 --k1 0.9 --b 0.4
    done
```

To evaluate NDCG@10 scores of BM25:

```bash
python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 dl19-passage \
  data/run.msmarco-v1-passage.bm25-default.dl19.txt
  
Results:
ndcg_cut_10           	all	0.5058
```

You can find the command line examples for full TREC DL datasets [here](https://castorini.github.io/pyserini/2cr/msmarco-v1-passage.html).

Similarly, you can find command lines for obtaining BM25 results on BEIR datasets [here](https://castorini.github.io/pyserini/2cr/beir.html).

In this repository, we use DL 2019 as an example. That is, we always re-rank `data/run.msmarco-v1-passage.bm25-default.dl19.txt` with LLMs.

--- 

## Cache Preference Queries for Future Runs

We will first run an `allpair` comparison to cache all preference queries.
Because this process can use be fully parallelized, we can use `accelerate` and batch inference to speed up the process.
After we get the cache, we can use it for future ranking experiments.

The following commands run on a machine with 4 x Tesla V100-SXM3-32GB GPUs:

```bash
model=meta-llama/Meta-Llama-3-8B-Instruct
method=allpair

# Run with ICL
accelerate launch --num_processes 4 run.py run \
--model_name_or_path $model --tokenizer_name_or_path $model --run_path data/run.msmarco-v1-passage.bm25-default.dl19.txt --ir_dataset_name msmarco-passage/trec-dl-2019 \
--save_path outputs/run.pairwise-icl.$method.${model#*/}.txt --hits 100 --query_length 32 --passage_length 128 \
--scoring generation --device cuda --temperature 0.0 --enable_ddp --enable_icl --icl_num 1 \
pairwise --method $method --k 100 --batch_size 8

# Run without ICL
accelerate launch --num_processes 4 run.py run \
--model_name_or_path $model --tokenizer_name_or_path $model --run_path data/run.msmarco-v1-passage.bm25-default.dl19.txt --ir_dataset_name msmarco-passage/trec-dl-2019 \
--save_path outputs/run.pairwise-icl.$method.${model#*/}.txt --hits 100 --query_length 32 --passage_length 128 \
--scoring generation --device cuda --temperature 0.0 --enable_ddp \
pairwise --method $method --k 100 --batch_size 8
```

## Calibration

After finishing the `allpair` comparison, we can calibrate the cached logits in each preference query.

```bash
python calibrate.py --model_dir "outputs/run.pairwise*preference_matrix"
```

## Pairwise Ranking with Sorting

The calibration and ICL ideally do not require `allpair` comparison.
Because we usually need to run the pairwise ranking multiple times, caching the preference queries can save a lot of time.
The sorted-based ranking with comparisons on the fly is possible, but it can still not be implemented in this repo.

```bash
model=meta-llama/Meta-Llama-3-8B-Instruct
for method in bubblesort heapsort
do

# Run with ICL
python run.py run \
--model_name_or_path $model --tokenizer_name_or_path $model --run_path data/run.msmarco-v1-passage.bm25-default.dl19.txt --ir_dataset_name msmarco-passage/trec-dl-2019 \
--save_path outputs/run.pairwise-icl.$method.${model#*/}.txt --hits 100 --query_length 32 --passage_length 128 \
--scoring generation --device cuda --temperature 0.0 --use_preference_cache  \
pairwise --method $method --k 100

# Run without ICL
python run.py run \
--model_name_or_path $model --tokenizer_name_or_path $model --run_path data/run.msmarco-v1-passage.bm25-default.dl19.txt --ir_dataset_name msmarco-passage/trec-dl-2019 \
--save_path outputs/run.pairwise.$method.${model#*/}.txt --hits 100 --query_length 32 --passage_length 128 \
--scoring generation --device cuda --temperature 0.0 --use_preference_cache  \
pairwise --method $method --k 100

# Run without ICL and without calibration
python run.py run \
--model_name_or_path $model --tokenizer_name_or_path $model --run_path data/run.msmarco-v1-passage.bm25-default.dl19.txt --ir_dataset_name msmarco-passage/trec-dl-2019 \
--save_path outputs/run.pairwise.$method.${model#*/}.txt --hits 100 --query_length 32 --passage_length 128 \
--scoring generation --device cuda --temperature 0.0 --use_preference_cache --use_ori_preference \
pairwise --method $method --k 100

done

```

## Aggregation

The `agg/aggregate_pyserini.py` script can aggregate ranking list files in pyserini format. 
The `docid_map_path` is the path to the preference matrix file, which contains the `docid-map_q-*.json`.
These files map the docid to the preference query index, which should be consistent within the same run.

`--input` is the path to the ranking list files, which should be in pyserini format. This can be a glob pattern to match multiple files.

`--output` is the path to the output file, which is also in pyserini format.

```bash
python agg/aggregate_pyserini.py --input "outputs/run.pairwise-icl.*.txt" \
--output "outputs/run.agg.txt" \
--docid_map_path "outputs/run.pairwise-icl.allpair.Meta-Llama-3-8B-Instruct_preference_matrix" 
```

## Evaluation

This Python script can evaluate the ranking list file in pyserini format.
`--log_file` is the path to the ranking list file, which should be in pyserini format. This can be a glob pattern to match multiple files.
It uses `joblib` to parallelize the evaluation process if there are multiple files.

```bash
python eval_fast.py --log_file "outputs/run.*.txt" \
--output_file "outputs/ndcg_scores.csv"
```

---
If you used our code for your research, please consider to cite our paper:

```text
@article{zeng2024llm,
  title={LLM-RankFusion: Mitigating Intrinsic Inconsistency in LLM-based Ranking},
  author={Zeng, Yifan and Tendolkar, Ojas and Baartmans, Raymond and Wu, Qingyun and Wang, Huazheng and Chen, Lizhong},
  journal={arXiv preprint arXiv:2406.00231},
  year={2024}
}
```
