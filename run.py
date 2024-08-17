import logging
import os
import ir_datasets
from accelerate import PartialState, Accelerator
from accelerate.utils import gather_object
from pyserini.search.lucene import LuceneSearcher
from pyserini.search._base import get_topics
from rankers.rankers import SearchResult
from rankers.pointwise import PointwiseLlmRanker, MonoT5LlmRanker
from rankers.setwise import SetwiseLlmRanker, OpenAiSetwiseLlmRanker
from rankers.pairwise import PairwiseLlmRanker, DuoT5LlmRanker, OpenAiPairwiseLlmRanker, ReasoningPairwiseLlmRanker
from rankers.listwise import OpenAiListwiseLlmRanker, ListwiseLlmRanker
from tqdm import tqdm
import argparse
import sys
import json
import time
import random
logger = logging.getLogger(__name__)


def parse_args(parser, commands):
    # Divide argv by commands
    split_argv = [[]]
    for c in sys.argv[1:]:
        if c in commands.choices:
            split_argv.append([c])
        else:
            split_argv[-1].append(c)
    # Initialize namespace
    args = argparse.Namespace()
    for c in commands.choices:
        setattr(args, c, None)
    # Parse each command
    parser.parse_args(split_argv[0], namespace=args)  # Without command
    for argv in split_argv[1:]:  # Commands
        n = argparse.Namespace()
        setattr(args, argv[0], n)
        parser.parse_args(argv, namespace=n)
    return args


def write_run_file(path, results, tag):
    with open(path, 'w') as f:
        for qid, _, ranking in results:
            rank = 1
            for doc in ranking:
                docid = doc.docid
                score = doc.score
                f.write(f"{qid}\tQ0\t{docid}\t{rank}\t{score}\t{tag}\n")
                rank += 1


def main(args):
    if os.path.exists(args.run.save_path):
        print(f'Run file {args.run.save_path} already exists. Skipping.')
        return
    distributed_state = None
    if args.pointwise:
        if 'monot5' in args.run.model_name_or_path:
            ranker = MonoT5LlmRanker(model_name_or_path=args.run.model_name_or_path,
                                     tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                     device=args.run.device,
                                     cache_dir=args.run.cache_dir,
                                     method=args.pointwise.method,
                                     batch_size=args.pointwise.batch_size)
        else:
            ranker = PointwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                        tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                        device=args.run.device,
                                        cache_dir=args.run.cache_dir,
                                        method=args.pointwise.method,
                                        batch_size=args.pointwise.batch_size)

    elif args.setwise:
        if args.run.openai_key:
            ranker = OpenAiSetwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                            api_key=args.run.openai_key,
                                            num_child=args.setwise.num_child,
                                            method=args.setwise.method,
                                            k=args.setwise.k)
        else:
            ranker = SetwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                      tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                      device=args.run.device,
                                      cache_dir=args.run.cache_dir,
                                      num_child=args.setwise.num_child,
                                      scoring=args.run.scoring,
                                      method=args.setwise.method,
                                      num_permutation=args.setwise.num_permutation,
                                      do_sample=args.run.do_sample,
                                      temperature=args.run.temperature,
                                      k=args.setwise.k)

    elif args.pairwise:
        if args.pairwise.method != 'allpair':
            args.pairwise.batch_size = 2
            logger.info(f'Setting batch_size to 2.')

        distributed_state = PartialState()

        if args.run.openai_key:
            ranker = OpenAiPairwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                             api_key=args.run.openai_key,
                                             method=args.pairwise.method,
                                             k=args.pairwise.k,
                                             args=args.run)

        elif 'duot5' in args.run.model_name_or_path:
            ranker = DuoT5LlmRanker(model_name_or_path=args.run.model_name_or_path,
                                    tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                    device=args.run.device,
                                    cache_dir=args.run.cache_dir,
                                    method=args.pairwise.method,
                                    batch_size=args.pairwise.batch_size,
                                    k=args.pairwise.k)
        elif args.pairwise.reasoning:
            ranker = ReasoningPairwiseLlmRanker(
                model_name_or_path=args.run.model_name_or_path,
                method=args.pairwise.method,
                batch_size=args.pairwise.batch_size,
                cache_dir=None,
                args=args.run,
                distributed_state=distributed_state)

        else:
            ranker = PairwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                       tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                       device=args.run.device,
                                       cache_dir=args.run.cache_dir,
                                       method=args.pairwise.method,
                                       batch_size=args.pairwise.batch_size,
                                       k=args.pairwise.k,
                                       do_sample=args.run.do_sample,
                                       temperature=args.run.temperature,
                                       args=args.run,
                                       distributed_state=distributed_state)

    elif args.listwise:
        if args.run.openai_key:
            ranker = OpenAiListwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                             api_key=args.run.openai_key,
                                             window_size=args.listwise.window_size,
                                             step_size=args.listwise.step_size,
                                             num_repeat=args.listwise.num_repeat)
        else:
            ranker = ListwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                       tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                       device=args.run.device,
                                       cache_dir=args.run.cache_dir,
                                       window_size=args.listwise.window_size,
                                       step_size=args.listwise.step_size,
                                       scoring=args.run.scoring,
                                       num_repeat=args.listwise.num_repeat)
    else:
        raise ValueError('Must specify either --pointwise, --setwise, --pairwise or --listwise.')

    query_map = {}
    if args.run.ir_dataset_name is not None:
        dataset = ir_datasets.load(args.run.ir_dataset_name)
        gt_rel = {f"{qrel.query_id}-{qrel.doc_id}": qrel for qrel in tqdm(dataset.qrels_iter())}
        for query in dataset.queries_iter():
            qid = query.query_id
            text = query.text
            query_map[qid] = ranker.truncate(text, args.run.query_length)
        dataset = ir_datasets.load(args.run.ir_dataset_name)
        docstore = dataset.docs_store()
    else:
        topics = get_topics(args.run.pyserini_index+'-test')
        for topic_id in list(topics.keys()):
            text = topics[topic_id]['title']
            query_map[str(topic_id)] = ranker.truncate(text, args.run.query_length)
        docstore = LuceneSearcher.from_prebuilt_index(args.run.pyserini_index+'.flat')

    logger.info(f'Loading first stage run from {args.run.run_path}.')
    first_stage_rankings = []
    with open(args.run.run_path, 'r') as f:
        current_qid = None
        current_ranking = []
        for line in tqdm(f):
            qid, _, docid, _, score, _ = line.strip().split()
            if qid != current_qid:
                if current_qid is not None:
                    first_stage_rankings.append((current_qid, query_map[current_qid], current_ranking[:args.run.hits]))
                current_ranking = []
                current_qid = qid
            if len(current_ranking) >= args.run.hits:
                continue
            if args.run.ir_dataset_name is not None:
                text = docstore.get(docid).text
                if 'title' in dir(docstore.get(docid)):
                    text = f'{docstore.get(docid).title} {text}'
            else:
                data = json.loads(docstore.doc(docid).raw())
                text = data['text']
                if 'title' in data:
                    text = f'{data["title"]} {text}'
            text = ranker.truncate(text, args.run.passage_length)
            current_ranking.append(SearchResult(docid=docid, score=float(score), text=text))
        first_stage_rankings.append((current_qid, query_map[current_qid], current_ranking[:args.run.hits]))

    reranked_results = []
    total_comparisons = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    tic = time.time()
    with distributed_state.split_between_processes(first_stage_rankings) as first_stage_rankings_split:
        for qid, query, ranking in tqdm(first_stage_rankings_split):
            if args.run.shuffle_ranking is not None:
                if args.run.shuffle_ranking == 'random':
                    random.seed(args.run.seed)
                    random.shuffle(ranking)
                elif args.run.shuffle_ranking == 'inverse':
                    ranking = ranking[::-1]
                else:
                    raise ValueError(f'Invalid shuffle ranking method: {args.run.shuffle_ranking}.')
            reranked_results.append((qid, query, ranker.rerank(qid, query, ranking)))
            total_comparisons += ranker.total_compare
            total_prompt_tokens += ranker.total_prompt_tokens
            total_completion_tokens += ranker.total_completion_tokens

    toc = time.time()
    if not args.run.enable_ddp:
        out = gather_object([total_comparisons, total_prompt_tokens, total_completion_tokens])
        total_comparisons, total_prompt_tokens, total_completion_tokens = out
        reranked_results = gather_object(reranked_results)

        print(f'Avg comparisons: {total_comparisons/len(reranked_results)}')
        print(f'Avg prompt tokens: {total_prompt_tokens/len(reranked_results)}')
        print(f'Avg completion tokens: {total_completion_tokens/len(reranked_results)}')
        print(f'Avg time per query: {(toc-tic)/len(reranked_results)}')
        print("Positional Bias Statistics:", getattr(ranker, "positional_bias_stat", {}))


        os.makedirs(args.run.save_path.rsplit('/', 1)[0], exist_ok=True)
        with open(args.run.save_path.replace('.txt', '_stats.json'), 'w') as f:
            json.dump({'avg_comparisons': total_comparisons/len(reranked_results),
                       'avg_prompt_tokens': total_prompt_tokens/len(reranked_results),
                       'avg_completion_tokens': total_completion_tokens/len(reranked_results),
                       'avg_time_per_query': (toc-tic)/len(reranked_results),
                       'positional_bias_stat': getattr(ranker, "positional_bias_stat", {})}, f, indent=4)

        write_run_file(args.run.save_path, reranked_results, 'LLMRankers')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(title='sub-commands')

    run_parser = commands.add_parser('run')
    run_parser.add_argument('--run_path', type=str, help='Path to the first stage run file (TREC format) to rerank.')
    run_parser.add_argument('--save_path', type=str, help='Path to save the reranked run file (TREC format).')
    run_parser.add_argument('--model_name_or_path', type=str,
                            help='Path to the pretrained model or model identifier from huggingface.co/models')
    run_parser.add_argument('--tokenizer_name_or_path', type=str, default=None,
                            help='Path to the pretrained tokenizer or tokenizer identifier from huggingface.co/tokenizers')
    run_parser.add_argument('--ir_dataset_name', type=str, default=None)
    run_parser.add_argument('--pyserini_index', type=str, default=None)
    run_parser.add_argument('--hits', type=int, default=100)
    run_parser.add_argument('--query_length', type=int, default=128)
    run_parser.add_argument('--passage_length', type=int, default=128)
    run_parser.add_argument('--device', type=str, default='cuda')
    run_parser.add_argument('--cache_dir', type=str, default=None)
    run_parser.add_argument('--openai_key', type=str, default=None)
    run_parser.add_argument('--scoring', type=str, default='generation', choices=['generation', 'likelihood'])
    run_parser.add_argument('--shuffle_ranking', type=str, default=None, choices=['inverse', 'random'])
    run_parser.add_argument('--do_sample', action='store_true', default=False)
    run_parser.add_argument('--temperature', type=float, default=0.0)
    run_parser.add_argument('--num_rank_runs', type=int, default=1)
    run_parser.add_argument('--enable_icl', action='store_true', default=False)
    run_parser.add_argument("--icl_relevance_gap", type=int, choices=[1, 2, 3], default=1)
    run_parser.add_argument("--icl_num", type=int, default=2)
    run_parser.add_argument("--icl_flipgt", action='store_true', default=False)
    run_parser.add_argument("--icl_example_path", type=str, default="data/icl_examples.json")
    run_parser.add_argument("--use_preference_cache", action='store_true', default=False)
    run_parser.add_argument("--preference_matrix_output_path", type=str, default=None)
    run_parser.add_argument("--seed", type=int, default=123)
    run_parser.add_argument("--enable_ddp", action='store_true', default=False)
    run_parser.add_argument("--enable_quantization", action='store_true', default=False)
    run_parser.add_argument("--use_ori_preference",  action='store_true', default=False)

    pointwise_parser = commands.add_parser('pointwise')
    pointwise_parser.add_argument('--method', type=str, default='yes_no',
                                  choices=['qlm', 'yes_no'])
    pointwise_parser.add_argument('--batch_size', type=int, default=2)

    pairwise_parser = commands.add_parser('pairwise')
    pairwise_parser.add_argument('--reasoning', action='store_true', default=False)
    pairwise_parser.add_argument('--method', type=str, default='allpair',
                                 choices=['allpair', 'heapsort', 'bubblesort'])
    pairwise_parser.add_argument('--batch_size', type=int, default=2)
    pairwise_parser.add_argument('--k', type=int, default=10)

    setwise_parser = commands.add_parser('setwise')
    setwise_parser.add_argument('--num_child', type=int, default=3)
    setwise_parser.add_argument('--method', type=str, default='heapsort',
                                choices=['heapsort', 'bubblesort'])
    setwise_parser.add_argument('--k', type=int, default=10)
    setwise_parser.add_argument('--num_permutation', type=int, default=1)

    listwise_parser = commands.add_parser('listwise')
    listwise_parser.add_argument('--window_size', type=int, default=3)
    listwise_parser.add_argument('--step_size', type=int, default=1)
    listwise_parser.add_argument('--num_repeat', type=int, default=1)

    args = parse_args(parser, commands)

    if args.run.ir_dataset_name is not None and args.run.pyserini_index is not None:
        raise ValueError('Must specify either --ir_dataset_name or --pyserini_index, not both.')

    arg_dict = vars(args)
    if arg_dict['run'] is None or sum(arg_dict[arg] is not None for arg in arg_dict) != 2:
        raise ValueError('Need to set --run and can only set one of --pointwise, --pairwise, --setwise, --listwise')
    if args.run.num_rank_runs > 1:
        assert args.run.do_sample and args.run.temperature > 0, 'Must set --do_sample and --temperature for multiple runs.'
        base_save_path = args.run.save_path
        for i in range(args.run.num_rank_runs):
            args.run.save_path = base_save_path.replace('.txt', f'_{i}.txt')
            main(args)
    else:
        main(args)
