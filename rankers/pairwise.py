import json
import os
from glob import glob
from os.path import join
from typing import List, Tuple
import numpy as np
import pandas as pd
from openai import OpenAI
from openai.lib.azure import AzureOpenAI
from retry import retry

from .rankers import LlmRanker, SearchResult
from itertools import combinations
from collections import defaultdict
from tqdm.auto import tqdm
import copy
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoConfig, AutoTokenizer, AutoModelForCausalLM, \
    QuantoConfig
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding
import tiktoken
import openai
import time
import re
from joblib import Parallel, delayed
from autogen.oai import OpenAIWrapper


class LogitAdjuster(torch.nn.Module):
    def __init__(self, tokenizer, candidate_words):
        super(LogitAdjuster, self).__init__()
        self.tokenizer = tokenizer
        self.candidate_ids = [self.tokenizer.encode(word, add_special_tokens=False)[0] for word in candidate_words]

    def forward(self, logits):
        # keep only the logits for the candidate words, set the rest to -inf
        logits = torch.where(
            torch.tensor([[i in self.candidate_ids for i in range(logits.shape[-1])]]).to(logits.device),
            logits, torch.full_like(logits, -float('inf')))

        return logits


def custom_generate(model, input_ids, decoder_input_ids, attention_mask, max_length, logit_adjuster, candidate_ids):
    cur_len = input_ids.shape[-1]
    with torch.no_grad():
        while cur_len < max_length:
            if decoder_input_ids is not None:
                outputs = model(input_ids, decoder_input_ids=decoder_input_ids, attention_mask=attention_mask)
            else:
                outputs = model(input_ids, attention_mask=attention_mask)
            next_token_logits = outputs.logits[:, -1, :]

            # Adjust logits using custom module
            adjusted_logits = logit_adjuster(next_token_logits.unsqueeze(1)).squeeze(1)

            # Sample from the adjusted logits
            next_token = torch.multinomial(torch.nn.functional.softmax(adjusted_logits, dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # get logits for the candidate_ids
            logits_arr = adjusted_logits[:, candidate_ids]

            cur_len += 1

        return input_ids, logits_arr


class Text2TextGenerationDataset(Dataset):
    def __init__(self, data: List[str], tokenizer: T5Tokenizer):
        self.data = tokenizer(data)

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, item):
        return {'input_ids': self.data['input_ids'][item],
                'attention_mask': self.data['attention_mask'][item]}


class PairwiseLlmRanker(LlmRanker):
    def __init__(self, model_name_or_path,
                 tokenizer_name_or_path,
                 device,
                 method="allpair",
                 batch_size=2,
                 k=10,
                 cache_dir=None,
                 temperature=0.0,
                 do_sample=False,
                 args=None,
                 distributed_state=None):
        self.model_name_or_path = model_name_or_path
        self.temperature = temperature
        self.do_sample = do_sample
        self.device = device
        self.method = method
        self.batch_size = batch_size
        self.k = k
        self.prompt = """Given a query "{query}", which of the following two passages is more relevant to the query?

Passage A: "{doc1}"

Passage B: "{doc2}"

Output Passage A or Passage B:"""
        self.args = args

        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0
        self.positional_bias_stat = {"AB": 0, "AA": 0, "BB": 0, "BA": 0, "N/A": 0}
        if self.args.use_preference_cache:
            # load preference matrix
            self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it",
                                                           cache_dir=cache_dir)  # placeholder tokenizer, not used
            self.preference_matrix, self.docid_map = {}, {}
            preference_matrix_output_path = self.args.preference_matrix_output_path \
                if self.args.preference_matrix_output_path is not None else \
                self.args.save_path.replace(".txt", "_preference_matrix").replace(
                "heapsort", "allpair").replace("bubblesort", "allpair").replace("quicksort", "allpair")
            if self.args.use_ori_preference:
                pref_file_list = sorted([[int(re.search(r"q-(\d+)_wocal\.npy", i).group(1)), i]
                                         for i in glob(join(preference_matrix_output_path, "q-*_wocal.npy"))],
                                        key=lambda x: x[0])
            else:
                pref_file_list = sorted([[int(re.search(r"q-(\d+)_fix\.npy", i).group(1)), i]
                                         for i in glob(join(preference_matrix_output_path, "q-*_fix.npy"))],
                                        key=lambda x: x[0])
            for qid, file in pref_file_list:
                self.preference_matrix[int(qid)] = np.load(file)

            docmap_file_list = sorted([[int(re.search(r"q-(\d+)\.json", i).group(1)), i]
                                       for i in glob(join(preference_matrix_output_path, "docid-map_q-*.json"))],
                                      key=lambda x: x[0])
            for qid, file in docmap_file_list:
                with open(file, "r") as f:
                    self.docid_map[qid] = json.load(f)

            if len(self.preference_matrix) == 0:
                print("Note found file in ", preference_matrix_output_path)
                raise FileNotFoundError("Preference matrix not found, rerank from scratch!!!")

            return

        self.config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        if self.config.model_type == 't5':
            self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name_or_path
                                                         if tokenizer_name_or_path is not None else
                                                         model_name_or_path, cache_dir=cache_dir)
            print("Load T5 model: ", model_name_or_path)
            if not self.args.enable_ddp:
                self.llm = T5ForConditionalGeneration.from_pretrained(model_name_or_path,
                                                                      device_map='auto',
                                                                      # max_memory={0: "24GiB", 1: "24GiB", 2: "24GiB", 3: "24GiB", "cpu": "0.5GiB"},
                                                                      torch_dtype=torch.float16,
                                                                      cache_dir=cache_dir)
            else:
                self.llm = T5ForConditionalGeneration.from_pretrained(model_name_or_path,
                                                                      torch_dtype=torch.float16,
                                                                      cache_dir=cache_dir)
                self.llm.to(distributed_state.device)
            self.decoder_input_ids = self.tokenizer.encode("<pad> Passage",
                                                           return_tensors="pt",
                                                           add_special_tokens=False).to(self.llm.device)
            self.decoder_input_ids = self.decoder_input_ids.repeat(self.batch_size, 1)
        elif any(chat_keyword in self.model_name_or_path for chat_keyword in ["it", "chat", "Instruct", "vicuna"]):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
            self.tokenizer.use_default_system_prompt = False
            if 'vicuna' and 'v1.5' in model_name_or_path:
                self.tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\\'s questions.' %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 %}{{ system_message }}{% endif %}{% if message['role'] == 'user' %}{{ ' USER: ' + message['content'].strip() }}{% elif message['role'] == 'assistant' %}{{ ' ASSISTANT: ' + message['content'].strip() + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ ' ASSISTANT:' }}{% endif %}"

            self.tokenizer.pad_token = "<|eot_id|>" if "Llama-3" in self.model_name_or_path else "[PAD]"
            self.tokenizer.padding_side = "left"

            if not self.args.enable_ddp:
                self.llm = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                                device_map='auto',  # auto sequential
                                                                # max_memory={0: "65GiB", 1: "65GiB", "cpu": "0.5GiB"},
                                                                torch_dtype=torch.float16 if device == 'cuda'
                                                                else torch.float32,
                                                                cache_dir=cache_dir).eval()
            else:
                self.llm = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                                torch_dtype=torch.float16 if device == 'cuda'
                                                                else torch.float32,
                                                                cache_dir=cache_dir).eval()
                self.llm.to(distributed_state.device)
        else:
            print("Using new model type: ", self.config.model_type)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
            self.llm = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                            device_map='sequential',
                                                            torch_dtype=torch.float16 if device == 'cuda'
                                                            else torch.float32,
                                                            cache_dir=cache_dir).eval()

        self.logit_adjuster = LogitAdjuster(self.tokenizer, ["A", "B"])
        self.create_icl(args)

    def create_icl(self, args):
        if self.args.enable_icl:
            icl_examples = pd.read_json(self.args.icl_example_path)
            self.icl_message = []
            for i in range(0, self.args.icl_num):
                self.icl_message += [
                    {"role": "user",
                     "content": self.prompt.format(
                         query=icl_examples.iloc[0].query_text,
                         doc1=icl_examples[icl_examples["relevance"] == 3].iloc[i].passage_text,
                         doc2=icl_examples[icl_examples["relevance"] == 3 - args.icl_relevance_gap].iloc[
                             i].passage_text)
                     },
                    {"role": "assistant", "content": "Passage: A" if not self.args.icl_flipgt else "Passage: B"},
                    {"role": "user",
                     "content": self.prompt.format(
                         query=icl_examples.iloc[0].query_text,
                         doc1=icl_examples[icl_examples["relevance"] == 3 - args.icl_relevance_gap].iloc[
                             i].passage_text,
                         doc2=icl_examples[icl_examples["relevance"] == 3].iloc[i].passage_text)
                     },
                    {"role": "assistant", "content": "Passage: B" if not self.args.icl_flipgt else "Passage: A"},
                ]
        else:
            self.icl_message = []

    def compare(self, qid, query: str, docs_id: List, docs: List):
        self.total_compare += 1
        doc1, doc2 = docs[0], docs[1]
        if self.args.use_preference_cache:
            if qid in self.preference_matrix.keys():
                cmp_res_AB = self.preference_matrix[qid][
                    self.docid_map[qid][docs_id[0]], self.docid_map[qid][docs_id[1]]]
                cmp_res_BA = self.preference_matrix[qid][
                    self.docid_map[qid][docs_id[1]], self.docid_map[qid][docs_id[0]]]
                return ["Passage A" if cmp_res_AB else "Passage B", "Passage A" if cmp_res_BA else "Passage B"]

        input_texts = [self.prompt.format(query=query, doc1=doc1, doc2=doc2),
                       self.prompt.format(query=query, doc1=doc2, doc2=doc1)]
        if self.config.model_type == 't5':
            if len(self.icl_message) > 0:
                input_texts[0] = "\n\n".join(
                    [self.icl_message[i]["content"] + ("<pad> Passage "
                             if "flan" in self.model_name_or_path
                             else " Passage ") + self.icl_message[i + 1]["content"].split()[-1]
                     for i in range(0, len(self.icl_message), 2)]) + "\n\n" + input_texts[0]
                input_texts[1] = "\n\n".join(
                    [self.icl_message[i]["content"] + ("<pad> Passage "
                             if "flan" in self.model_name_or_path
                             else " Passage ") + self.icl_message[i + 1]["content"].split()[-1]
                     for i in range(0, len(self.icl_message), 2)]) + "\n\n" + input_texts[1]

            input_ids = self.tokenizer(input_texts,
                                       padding='longest',
                                       return_tensors="pt").input_ids.to(self.llm.device)

            self.total_prompt_tokens += input_ids.shape[0] * input_ids.shape[1]

            output_ids = self.llm.generate(input_ids,
                                           decoder_input_ids=getattr(self, "decoder_input_ids", None),
                                           temperature=self.temperature, do_sample=self.do_sample,
                                           max_new_tokens=1)

            self.total_completion_tokens += output_ids.shape[0] * output_ids.shape[1]

            output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        elif any(chat_keyword in self.model_name_or_path for chat_keyword in ["it", "chat", "Instruct", "vicuna"]):
            conversation0 = self.icl_message + [{"role": "user", "content": input_texts[0]}]
            conversation1 = self.icl_message + [{"role": "user", "content": input_texts[1]}]

            prompt0 = self.tokenizer.apply_chat_template(conversation0, tokenize=False, add_generation_prompt=True)
            prompt0 += " Passage:"
            prompt1 = self.tokenizer.apply_chat_template(conversation1, tokenize=False, add_generation_prompt=True)
            prompt1 += " Passage:"

            input_ids = self.tokenizer([prompt0, prompt1], return_tensors="pt").input_ids.to(self.device)
            self.total_prompt_tokens += input_ids.shape[0] * input_ids.shape[1]

            output_ids = self.llm.generate(input_ids,
                                           temperature=self.temperature, do_sample=self.do_sample,
                                           max_new_tokens=1)  # The max_new_tokens=1 might be a bug in llama

            self.total_completion_tokens += output_ids.shape[0] * output_ids.shape[1]

            output0 = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:],
                                            skip_special_tokens=True).strip().upper()
            output1 = self.tokenizer.decode(output_ids[1][input_ids.shape[1]:],
                                            skip_special_tokens=True).strip().upper()
            return [f'Passage {output0}', f'Passage {output1}']
        else:
            input_ids = self.tokenizer(input_texts,
                                       padding='longest',
                                       return_tensors="pt").input_ids.to(self.llm.device)
            output_ids = self.llm.generate(input_ids,
                                           temperature=self.temperature, do_sample=self.do_sample,
                                           max_new_tokens=1)
            output = self.tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:],
                                                 skip_special_tokens=True)

        return output

    def heapify(self, arr, n, i):
        # Find largest among root and children
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2
        if l < n and arr[l] > arr[i]:
            largest = l

        if r < n and arr[r] > arr[largest]:
            largest = r

        # If root is not largest, swap with largest and continue heapifying
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            self.heapify(arr, n, largest)

    def heapSort(self, arr, k):
        n = len(arr)
        ranked = 0
        # Build max heap
        for i in range(n // 2, -1, -1):
            self.heapify(arr, n, i)
        for i in range(n - 1, 0, -1):
            # Swap
            arr[i], arr[0] = arr[0], arr[i]
            ranked += 1
            if ranked == k:
                break
            # Heapify root element
            self.heapify(arr, i, 0)

    def rerank(self, qid, query: str, ranking: List[SearchResult]) -> List[SearchResult]:
        original_ranking = copy.deepcopy(ranking)
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0
        if self.method == "allpair":
            preference_matrix_output_path = self.args.save_path.replace(".txt", "_preference_matrix")
            if os.path.exists(join(preference_matrix_output_path, f'q-{qid}.npy')):
                print("Preference matrix exists, skip rerank, final ranking result not trustful!!!")
                return []
            preference_matrix = np.zeros((len(ranking), len(ranking)), dtype=np.bool_)
            preference_logit = np.zeros((len(ranking), len(ranking), 2), dtype=np.float16)
            docid_map = {doc.docid: i for i, doc in enumerate(ranking)}
            doc_pairs = list(combinations(ranking, 2))
            allpairs = []

            if any(chat_keyword in self.model_name_or_path for chat_keyword in ["it", "chat", "Instruct", "vicuna"]):
                for doc1, doc2 in tqdm(doc_pairs):
                    conversation0 = self.icl_message + [{"role": "user", "content":
                        self.prompt.format(query=query, doc1=doc1.text, doc2=doc2.text)}]
                    prompt0 = self.tokenizer.apply_chat_template(conversation0, tokenize=False,
                                                                 add_generation_prompt=True)
                    prompt0 += " Passage:"
                    allpairs.append(prompt0)
                    conversation1 = self.icl_message + [{"role": "user", "content":
                        self.prompt.format(query=query, doc1=doc2.text, doc2=doc1.text)}]
                    prompt1 = self.tokenizer.apply_chat_template(conversation1, tokenize=False,
                                                                 add_generation_prompt=True)
                    prompt1 += " Passage:"
                    allpairs.append(prompt1)
            elif "gpt" not in self.model_name_or_path:
                for doc1, doc2 in tqdm(doc_pairs):
                    input_texts = [self.prompt.format(query=query, doc1=doc1.text, doc2=doc2.text),
                                   self.prompt.format(query=query, doc1=doc2.text, doc2=doc1.text)]
                    if len(self.icl_message) > 0:
                        input_texts[0] = "\n\n".join(
                            [self.icl_message[i]["content"] + ("<pad> Passage "
                             if "flan" in self.model_name_or_path
                             else " Passage ") + self.icl_message[i + 1]["content"].split()[-1]
                             for i in range(0, len(self.icl_message), 2)]) + "\n\n" + input_texts[0]
                        input_texts[1] = "\n\n".join(
                            [self.icl_message[i]["content"] + ("<pad> Passage "
                             if "flan" in self.model_name_or_path
                             else " Passage ") + self.icl_message[i + 1]["content"].split()[-1]
                             for i in range(0, len(self.icl_message), 2)]) + "\n\n" + input_texts[1]
                    allpairs.append(input_texts[0])
                    allpairs.append(input_texts[1])
            else:
                for doc1, doc2 in tqdm(doc_pairs):
                    allpairs.append(self.prompt.format(query=query, doc1=doc1.text, doc2=doc2.text))
                    allpairs.append(self.prompt.format(query=query, doc1=doc2.text, doc2=doc1.text))

            if self.model_name_or_path.startswith("gpt"):
                outputs = Parallel(backend='threading', n_jobs=6)(
                    delayed(self._get_response)(i, history=self.icl_message) for i in tqdm(allpairs))
                outputs_logits = [i[1] for i in outputs]
                outputs = [i[0] for i in outputs]
            else:
                allpairs_dataset = Text2TextGenerationDataset(allpairs, self.tokenizer)

                loader = DataLoader(
                    allpairs_dataset,
                    batch_size=self.batch_size,
                    collate_fn=DataCollatorWithPadding(
                        self.tokenizer,
                        max_length=512,
                        padding='longest',
                    ),
                    shuffle=False,
                    drop_last=False,
                    num_workers=1
                )

                outputs = []
                outputs_logits = []
                for batch_inputs in tqdm(loader):
                    self.total_compare += 1
                    self.total_prompt_tokens += batch_inputs['input_ids'].shape[0] * batch_inputs['input_ids'].shape[1]

                    if getattr(self, "decoder_input_ids", None) is not None:
                        decoder_input_ids = self.decoder_input_ids if self.decoder_input_ids.shape[0] == len(
                            batch_inputs['input_ids']) else self.decoder_input_ids[:len(batch_inputs['input_ids']),
                                                            :]  # last batch might be smaller
                    else:
                        decoder_input_ids = None

                    # batch_outputs = self.llm.generate(batch_inputs['input_ids'].to(self.llm.device),
                    #                                   decoder_input_ids=decoder_input_ids,
                    #                                   attention_mask=batch_inputs['attention_mask'].to(self.llm.device),
                    #                                   max_new_tokens=1)

                    batch_outputs, logits_arr = custom_generate(
                        self.llm, batch_inputs['input_ids'].to(self.llm.device),
                        decoder_input_ids=decoder_input_ids,
                        attention_mask=batch_inputs['attention_mask'].to(self.llm.device),
                        max_length=batch_inputs['input_ids'].shape[1] + 1, logit_adjuster=self.logit_adjuster,
                        candidate_ids=[self.tokenizer.encode(word, add_special_tokens=False)[0] for word in ["A", "B"]]
                    )

                    self.total_completion_tokens += batch_outputs.shape[0] * batch_outputs.shape[1]
                    outputs.extend(batch_outputs.cpu().numpy())
                    outputs_logits.extend(logits_arr.cpu().numpy())

                outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            scores = defaultdict(float)
            for i in range(0, len(outputs), 2):
                doc1, doc2 = doc_pairs[i // 2]
                output1 = outputs[i][-1]
                output2 = outputs[i + 1][-1]

                stat_key = (output1[-1].upper() + output2[-1].upper()) if (
                        len(output1) > 0 and len(output2) > 0) else "N/A"
                if stat_key in self.positional_bias_stat.keys():
                    self.positional_bias_stat[stat_key] += 1
                else:
                    self.positional_bias_stat["N/A"] += 1

                if ("A" in output1.upper()) and ("B" in output2.upper()):
                    scores[doc1.docid] += 1
                elif ("B" in output1.upper()) and ("A" in output2.upper()):
                    scores[doc2.docid] += 1
                else:  # conflict
                    scores[doc1.docid] += 0.5
                    scores[doc2.docid] += 0.5

                if "A" in output1.upper():
                    preference_matrix[docid_map[doc1.docid], docid_map[doc2.docid]] = True
                # elif "B" in output1.upper():
                #     preference_matrix[docid_map[doc2.docid], docid_map[doc1.docid]] = True

                if "A" in output2.upper():
                    preference_matrix[docid_map[doc2.docid], docid_map[doc1.docid]] = True
                # elif "B" in output2.upper():
                #     preference_matrix[docid_map[doc1.docid], docid_map[doc2.docid]] = True

                preference_logit[docid_map[doc1.docid], docid_map[doc2.docid], 0] = outputs_logits[i][0]
                preference_logit[docid_map[doc1.docid], docid_map[doc2.docid], 1] = outputs_logits[i][1]
                preference_logit[docid_map[doc2.docid], docid_map[doc1.docid], 0] = outputs_logits[i + 1][0]
                preference_logit[docid_map[doc2.docid], docid_map[doc1.docid], 1] = outputs_logits[i + 1][1]

            ranking = sorted([SearchResult(docid=docid, score=score, text=None) for docid, score in scores.items()],
                             key=lambda x: x.score, reverse=True)

            os.makedirs(preference_matrix_output_path, exist_ok=True)
            np.save(join(preference_matrix_output_path, f"q-{qid}_ori.npy"), preference_matrix)
            np.save(join(preference_matrix_output_path, f"q-{qid}_logit.npy"), preference_logit)
            with open(join(preference_matrix_output_path, f'docid-map_q-{qid}.json'), "w") as f:
                json.dump(docid_map, f)

        elif self.method == "heapsort":
            class ComparableDoc:
                def __init__(self, docid, text, ranker):
                    self.docid = docid
                    self.text = text
                    self.ranker = ranker

                def __gt__(self, other):
                    out = self.ranker.compare(int(qid), query, [self.docid, other.docid], [self, other])

                    stat_key = "".join([o[-1].upper() if len(o) > 0 else "" for o in out])
                    if stat_key in self.ranker.positional_bias_stat.keys():
                        self.ranker.positional_bias_stat[stat_key] += 1
                    else:
                        self.ranker.positional_bias_stat["N/A"] += 1

                    if ("passage a" in out[0].lower()) and ("passage b" in out[1].lower()):
                        return True
                    else:
                        return False

            arr = [ComparableDoc(docid=doc.docid, text=doc.text, ranker=self) for doc in ranking]
            self.heapSort(arr, self.k)
            ranking = [SearchResult(docid=doc.docid, score=-i, text=None) for i, doc in enumerate(reversed(arr))]

        #
        # elif self.method == "bubblesort":
        #     k = min(k, len(ranking))
        #     for i in range(k):
        #         current_ind = len(ranking) - 1
        #         while True:
        #             if current_ind == i:
        #                 break
        #             doc1 = ranking[current_ind]
        #             doc2 = ranking[current_ind - 1]
        #             output = self.compare(query, [doc1.text, doc2.text])
        #             if output[0] == "Passage A" and output[1] == "Passage B":
        #                 ranking[current_ind - 1], ranking[current_ind] = ranking[current_ind], ranking[current_ind - 1]
        #             current_ind -= 1
        elif self.method == "bubblesort":
            k = min(self.k, len(ranking))

            last_end = len(ranking) - 1
            for i in range(k):
                current_ind = last_end
                is_change = False
                while True:
                    if current_ind <= i:
                        break
                    doc1 = ranking[current_ind]
                    doc2 = ranking[current_ind - 1]
                    output = self.compare(int(qid), query, [doc1.docid, doc2.docid], [doc1.text, doc2.text])
                    if output[0] == "Passage A" and output[1] == "Passage B":
                        ranking[current_ind - 1], ranking[current_ind] = ranking[current_ind], ranking[current_ind - 1]

                        if not is_change:
                            is_change = True
                            if last_end != len(ranking) - 1:  # skip unchanged pairs at the bottom
                                last_end += 1
                    if not is_change:
                        last_end -= 1
                    current_ind -= 1
        else:
            raise NotImplementedError(f'Method {self.method} is not implemented.')

        results = []
        top_doc_ids = set()
        rank = 1
        for i, doc in enumerate(ranking[:self.k]):
            top_doc_ids.add(doc.docid)
            results.append(SearchResult(docid=doc.docid, score=-rank, text=None))
            rank += 1
        for doc in original_ranking:
            if doc.docid not in top_doc_ids:
                results.append(SearchResult(docid=doc.docid, score=-rank, text=None))
                rank += 1
        return results

    def truncate(self, text, length):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(text)[:length])


class ReasoningPairwiseLlmRanker(PairwiseLlmRanker):
    def __init__(self, model_name_or_path,
                 method="allpair",
                 batch_size=2,
                 cache_dir=None,
                 args=None,
                 distributed_state=None):
        self.model_name_or_path = model_name_or_path
        self.method = method
        self.batch_size = batch_size
        self.prompt = """Given a query "{query}", analyze the following two passages and provide step-by-step reasoning for which passage is more relevant to the query. Consider the following steps in your reasoning:

1. Examine each passage to determine how well it addresses the key terms and concepts.
2. Compare the relevance of each passage to the query.
3. Decide which passage is more relevant.

Please be very concise in reasoning. After your step-by-step reasoning, output the choice of passage in the following format.

Output: Passage <A|B>

Passage A: "{doc1}"

Passage B: "{doc2}"

"""
        self.args = args

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.tokenizer.use_default_system_prompt = False
        self.tokenizer.pad_token = "<|eot_id|>" if "Llama-3" in self.model_name_or_path else "[PAD]"
        self.tokenizer.padding_side = "left"
        if self.args.enable_quantization:
            quantization_config = QuantoConfig(weights="float8")
        else:
            quantization_config = None
        if self.args.enable_ddp:
            self.llm = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                            torch_dtype=torch.float16,
                                                            cache_dir=cache_dir,
                                                            quantization_config=quantization_config).eval()
            self.llm.to(distributed_state.device)

        else:
            self.llm = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                            device_map='auto',  # auto sequential
                                                            torch_dtype=torch.float16,
                                                            cache_dir=cache_dir,
                                                            quantization_config=quantization_config).eval()

    def rerank(self, qid, query: str, ranking: List[SearchResult]) -> List[SearchResult]:
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0
        if self.method == "allpair":
            preference_matrix_output_path = self.args.save_path.replace(".txt", "_output_text")
            if os.path.exists(join(preference_matrix_output_path, f'q-{qid}.json')):
                print("Output text exists, skip rerank, final ranking result not trustful!!!")
                return []
            doc_pairs = list(combinations(ranking, 2))
            allpairs = []

            if any(chat_keyword in self.model_name_or_path for chat_keyword in ["it", "chat", "Instruct", "vicuna"]):
                for doc1, doc2 in tqdm(doc_pairs):
                    conversation0 = [{"role": "user", "content":
                        self.prompt.format(query=query, doc1=doc1.text, doc2=doc2.text)}]
                    prompt0 = self.tokenizer.apply_chat_template(conversation0, tokenize=False,
                                                                 add_generation_prompt=True)
                    prompt0 += "Step-by-step reasoning:"
                    allpairs.append(prompt0)
                    conversation1 = [{"role": "user", "content":
                        self.prompt.format(query=query, doc1=doc2.text, doc2=doc1.text)}]
                    prompt1 = self.tokenizer.apply_chat_template(conversation1, tokenize=False,
                                                                 add_generation_prompt=True)
                    prompt1 += "Step-by-step reasoning:"
                    allpairs.append(prompt1)
            else:
                for doc1, doc2 in tqdm(doc_pairs):
                    allpairs.append(self.prompt.format(query=query, doc1=doc1.text, doc2=doc2.text))
                    allpairs.append(self.prompt.format(query=query, doc1=doc2.text, doc2=doc1.text))

            allpairs_dataset = Text2TextGenerationDataset(allpairs, self.tokenizer)

            loader = DataLoader(
                allpairs_dataset,
                batch_size=self.batch_size,
                collate_fn=DataCollatorWithPadding(
                    self.tokenizer,
                    max_length=512,
                    padding='longest',
                ),
                shuffle=False,
                drop_last=False,
                num_workers=4
            )

            outputs = []
            for batch_inputs in tqdm(loader):
                self.total_compare += 1
                self.total_prompt_tokens += batch_inputs['input_ids'].shape[0] * batch_inputs['input_ids'].shape[1]

                if getattr(self, "decoder_input_ids", None) is not None:
                    decoder_input_ids = self.decoder_input_ids if self.decoder_input_ids.shape[0] == len(
                        batch_inputs['input_ids']) else self.decoder_input_ids[:len(batch_inputs['input_ids']), :]
                    # last batch might be smaller
                else:
                    decoder_input_ids = None

                batch_outputs = self.llm.generate(batch_inputs['input_ids'].to(self.llm.device),
                                                  decoder_input_ids=decoder_input_ids,
                                                  temperature=self.args.temperature, do_sample=self.args.do_sample,
                                                  attention_mask=batch_inputs['attention_mask'].to(self.llm.device),
                                                  max_new_tokens=130)

                self.total_completion_tokens += batch_outputs.shape[0] * batch_outputs.shape[1]
                outputs.extend(batch_outputs.cpu().numpy())

                # debug
                # print(self.tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)[-1].split("assistant\n\nStep-by-step reasoning:")[-1].strip())

            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            outputs_text_collection = {}
            for i in range(0, len(outputs), 2):
                doc1, doc2 = doc_pairs[i // 2]  # i // 2
                outputs_text_collection[f"{doc1.docid}-{doc2.docid}"] = \
                    outputs[i].split("assistant\n\nStep-by-step reasoning:")[-1].strip()
                outputs_text_collection[f"{doc2.docid}-{doc1.docid}"] = \
                    outputs[i + 1].split("assistant\n\nStep-by-step reasoning:")[-1].strip()

            os.makedirs(preference_matrix_output_path, exist_ok=True)
            with open(join(preference_matrix_output_path, f"q-{qid}.json"), 'w') as fd:
                json.dump(outputs_text_collection, fd, ensure_ascii=False, indent=4)


class DuoT5LlmRanker(PairwiseLlmRanker):
    def compare(self, qid, query: str, docs: List[str]) -> bool:
        self.total_compare += 1
        self.prompt = 'Query: {query} Document0: {doc1} Document1: {doc2} Relevant:'

        inputs = [self.prompt.format(query=query, doc1=docs[0], doc2=docs[1]),
                  self.prompt.format(query=query, doc1=docs[1], doc2=docs[0])]
        inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt").to(self.llm.device)
        decode_ids = torch.full((2, 1),
                                self.llm.config.decoder_start_token_id,
                                dtype=torch.long, device=self.llm.device)

        self.total_prompt_tokens += inputs['input_ids'].shape[0] * inputs['input_ids'].shape[1]

        with torch.no_grad():
            logits = self.llm(input_ids=inputs['input_ids'],
                              attention_mask=inputs['attention_mask'],
                              decoder_input_ids=decode_ids).logits
            # 6136 and 1176 are the indexes of the tokens false and true in T5.
            batch_scores = logits[:, 0, [6136, 1176]]
            batch_scores = torch.nn.functional.softmax(batch_scores, dim=1)
            batch_probs = batch_scores[:, 1]
        return batch_probs[0] > batch_probs[1]

    def rerank(self, qid, query: str, ranking: List[SearchResult]) -> List[SearchResult]:
        original_ranking = copy.deepcopy(ranking)
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0
        if self.method == "heapsort":
            class ComparableDoc:
                def __init__(self, docid, text, ranker):
                    self.docid = docid
                    self.text = text
                    self.ranker = ranker

                def __gt__(self, other):
                    return self.ranker.compare(int(qid), query, [self.text, other.text])

            arr = [ComparableDoc(docid=doc.docid, text=doc.text, ranker=self) for doc in ranking]
            self.heapSort(arr, self.k)
            ranking = [SearchResult(docid=doc.docid, score=-i, text=None) for i, doc in enumerate(reversed(arr))]

        else:
            raise NotImplementedError(f'Method {self.method} is not implemented.')

        results = []
        top_doc_ids = set()
        rank = 1
        for i, doc in enumerate(ranking[:self.k]):
            top_doc_ids.add(doc.docid)
            results.append(SearchResult(docid=doc.docid, score=-rank, text=None))
            rank += 1
        for doc in original_ranking:
            if doc.docid not in top_doc_ids:
                results.append(SearchResult(docid=doc.docid, score=-rank, text=None))
                rank += 1
        return results


class OpenAiPairwiseLlmRanker(PairwiseLlmRanker):
    def __init__(self,
                 model_name_or_path,
                 api_key,
                 method="heapsort",
                 batch_size=2,
                 k=10,
                 args=None):
        self.args = args
        self.llm = self.model_name_or_path = model_name_or_path
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.method = method
        self.k = k
        self.total_compare = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.CHARACTERS = ["A", "B"]
        self.system_prompt = "You are RankGPT, an intelligent assistant specialized in selecting the most relevant passage from a pair of passages based on their relevance to the query. You should only output a single letter 'A' or 'B'."
        self.prompt = """Given a query "{query}", which of the following two passages is more relevant to the query?
        
Passage A: "{doc1}"

Passage B: "{doc2}"

Output A or B:"""
        self.client = OpenAIWrapper(config_list=[
            {
                "model": self.llm,
                "api_key": api_key,
                "api_type": "azure",
                "base_url": "https://yifan3.openai.azure.com/",
                "api_version": "2024-02-15-preview",
                "cache_seed": 123,
            }
        ])
        self.positional_bias_stat = {"AB": 0, "AA": 0, "BB": 0, "BA": 0, "N/A": 0}
        self.create_icl(args)

    def _get_batch_response(self, allpairs):
        # write request batch to a file, each line looks like `{"custom_id": "request-1", "method": "POST",
        # "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo", "messages": [{"role": "system",
        # "content": "You are a helpful assistant."}, {"role": "user", "content": "What is 2+2?"}]}}`
        with open("/tmp/batch_request.jsonl", "w") as fd:
            for i, input_text in enumerate(allpairs):
                messages = self.icl_message + [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": input_text}
                ]
                json_line = {"custom_id": f"request-{i}", "method": "POST", "url": "/v1/chat/completions",
                             "body": {"model": self.llm, "temperature": 0.0, "max_tokens": 5, "messages": messages}}
                fd.write(json.dumps(json_line, ensure_ascii=True) + "\n")
            fd.close()

        # upload the file to Azure Blob Storage
        input_file_id = self.client.files.create(file=open("/tmp/batch_request.jsonl", "rb"), purpose="assistants").id

        # create a batch
        response = self.client.batches.create(input_file_id=input_file_id, completion_window="24h",
                                              endpoint="/v1/chat/completions")
        while True:
            batch_data = self.client.batches.retrieve(batch_id=response.id)
            if batch_data.status == "completed":
                break
            elif batch_data.status == "failed":
                print(self.client.files.content(batch_data.error_file_id))
                raise Exception("Batch failed")
            time.sleep(5)

        batch_outputs = self.client.files.content(batch_data.output_file_id)

        # The per-line object of the batch output and error files: `{"id": "batch_req_wnaDys", "custom_id":
        # "request-2", "response": {"status_code": 200, "request_id": "req_c187b3", "body": {"id": "chatcmpl-9758Iw",
        # "object": "chat.completion", "created": 1711475054, "model": "gpt-3.5-turbo", "choices": [{"index": 0,
        # "message": {"role": "assistant", "content": "2 + 2 equals 4."}, "finish_reason": "stop"}],
        # "usage": {"prompt_tokens": 24, "completion_tokens": 15, "total_tokens": 39}, "system_fingerprint": null}},
        # "error": null}`

        outputs = []
        for line in batch_outputs.text.split("\n"):
            response = json.loads(line)
            output = response["response"]["body"]["choices"][0]["message"]["content"]
            output = self.parse_output(output)
            outputs.append(output)

        return outputs

    @retry((openai.APIError, openai.RateLimitError, openai.APITimeoutError), delay=2, backoff=2, max_delay=10)
    def _get_response(self, input_text, history=None):
        if history is None:
            history = []

        response = self.client.create(
            model=self.llm,
            messages=[{"role": "system", "content": self.system_prompt}] + history + [{"role": "user", "content": input_text}],
            temperature=0.0, max_tokens=5,
            logprobs=True, top_logprobs=5
        )
        self.total_completion_tokens += int(response.usage.completion_tokens)
        self.total_prompt_tokens += int(response.usage.prompt_tokens)

        output = response.choices[0].message.content
        output = self.parse_output(output)

        min_logprob = 50
        prob = [-50, -50]
        for token in response.choices[0].logprobs.content:
            if token.token.strip() in ["A", "B"]:
                for j in token.top_logprobs:
                    min_logprob = min(min_logprob, j.logprob)
                    if j.token.strip() == "A" and j.logprob > prob[0]:
                        prob[0] = j.logprob
                    elif j.token.strip() == "B" and j.logprob > prob[1]:
                        prob[1] = j.logprob
                break
        prob[0], prob[1] = max(prob[0], min_logprob) - 1, max(prob[1], min_logprob) - 1
        return output, prob

    def parse_output(self, output):
        if any(kw in output.lower() for kw in ["neither", "both", "similar", "equally"]):
            return "E"
        matches = re.findall(r"[:et]{1} ([A-B])", output, re.MULTILINE)
        if matches:
            output = matches[0][-1]
        elif output.strip() in self.CHARACTERS:
            pass
        elif output[-1] in self.CHARACTERS:
            output = output[-1]
        else:
            print(f"Unexpected output: {output}")
            output = "A"
        return output

    def compare(self, qid, query: str, docs_id: List, docs: List):
        self.total_compare += 1
        doc1, doc2 = docs[0], docs[1]
        if self.args.use_preference_cache:
            if qid in self.preference_matrix.keys():
                cmp_res_AB = self.preference_matrix[qid][
                    self.docid_map[qid][docs_id[0]], self.docid_map[qid][docs_id[1]]]
                cmp_res_BA = self.preference_matrix[qid][
                    self.docid_map[qid][docs_id[1]], self.docid_map[qid][docs_id[0]]]
                return ["Passage A" if cmp_res_AB else "Passage B", "Passage A" if cmp_res_BA else "Passage B"]
        input_texts = [self.prompt.format(query=query, doc1=doc1, doc2=doc2),
                       self.prompt.format(query=query, doc1=doc2, doc2=doc1)]

        return [f'Passage {self._get_response(input_texts[0], history=self.icl_message)}',
                f'Passage {self._get_response(input_texts[1], history=self.icl_message)}']

    def truncate(self, text, length):
        return self.tokenizer.decode(self.tokenizer.encode(text)[:length])
