"""
Dataset for stage 1 extractor

prepare() fn adapted from https://github.com/facebookresearch/multihop_dense_retrieval 

@author Tim Hartill

"""

import collections
import json
import random
import copy

import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from mdr_basic_tokenizer_and_utils import SimpleTokenizer, para_has_answer, match_answer_span, find_ans_span_with_char_offsets
from text_processing import is_whitespace, get_sentence_list

from utils import collate_tokens, get_para_idxs, consistent_bridge_format



def prepare(item, tokenizer, special_toks=["[SEP]", "[unused0]", "[unused1]", "[unused2]", "[unused3]"]):
    """
    tokenize the passages chains, add sentence start markers for SP sentence identification
    """
    def _process_p(para):
        """
        handle each para
        """
        title, sentence_spans = para["title"].strip(), para["sentence_spans"]
        # return "[unused1] " + title + " [unused1] " + text # mark title
        # return title + " " + text
        sents = get_sentence_list(para["text"], sentence_spans)
        pre_sents = []
        for idx, sent in enumerate(sents):
            pre_sents.append("[unused1] " + sent.strip())
        return title + " " + " ".join(pre_sents)
        # return " ".join(pre_sents)
    # mark passage boundary
    contexts = []
    for para in item["pos_paras"]:
        contexts.append(_process_p(para))
    context = " [SEP] ".join(contexts)

    doc_tokens = []  # TJH: ['word1', 'word2', ..]
    char_to_word_offset = []  # TJH: list with each char -> idx into doc_tokens
    prev_is_whitespace = True

    context = "yes no [unused0] [SEP] " + context  # ELECTRA tokenises yes, no to single tokens

    for c in context:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)

    sent_starts = []
    orig_to_tok_index = []
    tok_to_orig_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))

        if token in special_toks:
            if token == "[unused1]":
                sent_starts.append(len(all_doc_tokens))  # [sentence start idx -> subword idx]

            sub_tokens = [token]
        else:
            sub_tokens = tokenizer.tokenize(token)

        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)       # [ subword tok idx -> whole word token idx]
            all_doc_tokens.append(sub_token)  # [ sub word tokens ]

    item["context_processed"] = {
        "doc_tokens": doc_tokens,                     # [whole words]
        "char_to_word_offset": char_to_word_offset,   # [char idx -> whole word idx]
        "orig_to_tok_index": orig_to_tok_index,       # [whole word idx -> subword idx]
        "tok_to_orig_index": tok_to_orig_index,       # [ subword tok idx -> whole word token idx]
        "all_doc_tokens": all_doc_tokens,             # [ sub word tokens ]
        "context": context,                           # full context string    
        "sent_starts": sent_starts                    # [sentence start idx -> subword idx]
    }
    return item



class ExtractorDataset(Dataset):

    def __init__(self, args, tokenizer, train=False):
        self.data_path = args.predict_file
        samples = [json.loads(l) for l in tqdm(open(self.data_path).readlines())]
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_len
        self.max_q_len = args.max_q_len
        self.train = train
        self.simple_tok = SimpleTokenizer()
        self.data = []  # Each alternate sample will be a blank placeholder denoting a negative for preceding positive
        for sample in samples:
            if sample["question"].endswith("?"):
                sample["question"] = sample["question"][:-1]
            consistent_bridge_format(sample)    
            #TODO add para label in __get_item__
            self.data.append(sample)
            self.data.append({})  # dummy entry for neg example - construct actual neg from data[index-1] 
            #TODO How to have fixed set of eval queries for eval if building on the fly? Just use fixed "bridge" order?


        if train:
            self.qid2gold = collections.defaultdict(list) # idx of q + gold sp in self.data
            self.qid2neg = collections.defaultdict(list)
            for item in retriever_outputs:
                if item["question"].endswith("?"):
                    item["question"] = item["question"][:-1]

                sp_sent_labels = []
                sp_gold = []
                if not self.no_sent_label:
                    for sp in item["sp"]:
                        for _ in sp["sp_sent_ids"]:
                            sp_gold.append([sp["title"], _])
                        for idx in range(len(sp["sents"])):
                            sp_sent_labels.append(int(idx in sp["sp_sent_ids"]))

                question_type = item["type"]
                self.data.append({
                    "question": item["question"],
                    "passages": item["sp"], 
                    "label": 1,
                    "qid": item["_id"],
                    "gold_answer": item["answer"],
                    "sp_sent_labels": sp_sent_labels,
                    "ans_covered": 1, # includes partial chains.
                    "sp_gold": sp_gold
                })
                self.qid2gold[item["_id"]].append(len(self.data) - 1)

                sp_titles = set([_["title"] for _ in item["sp"]])
                if question_type == "bridge":
                    ans_titles = set([p["title"] for p in item["sp"] if para_has_answer(item["answer"], "".join(p["sents"]), self.simple_tok)])
                else:
                    ans_titles = set()
                # top ranked negative chains
                ds_count = 0 # track how many distant supervised chain to use
                ds_limit = 5
                for chain in item["candidate_chains"]:
                    chain_titles = [c["title"] for c in chain]
                    if set(chain_titles) == sp_titles: 
                        continue
                    if question_type == "bridge":
                        answer_covered = int(len(set(chain_titles) & ans_titles) > 0)  #if any pred para is an sp title that has the answer in it
                        ds_count += answer_covered
                    else:
                        answer_covered = 0
                    self.data.append({
                        "question": item["question"],
                        "passages": chain,
                        "label": 0,
                        "qid": item["_id"],
                        "gold_answer": item["answer"],
                        "ans_covered": answer_covered,
                        "sp_gold": sp_gold
                    })
                    self.qid2neg[item["_id"]].append(len(self.data) - 1)
        else:
            for item in retriever_outputs:
                if item["question"].endswith("?"):
                    item["question"] = item["question"][:-1]

                # for validation, add target predictions
                sp_titles = set([_["title"] for _ in item["sp"]]) if "sp" in item else None
                gold_answer = item.get("answer", [])
                sp_gold = []
                if "sp" in item:
                    for sp in item["sp"]:
                        for _ in sp["sp_sent_ids"]:
                            sp_gold.append([sp["title"], _])

                chain_seen = set()
                for chain in item["candidate_chains"]:
                    chain_titles = [_["title"] for _ in chain]


                    if sp_titles:
                        label = int(set(chain_titles) == sp_titles)
                    else:
                        label = -1
                    self.data.append({
                        "question": item["question"],
                        "passages": chain,
                        "label": label,
                        "qid": item["_id"],
                        "gold_answer": gold_answer,
                        "sp_gold": sp_gold
                    })

        print(f"Data size {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = prepare(self.data[index], self.tokenizer) 
        context_ann = item["context_processed"]  #TJH context_processed added to item in prepare
        q_toks = self.tokenizer.tokenize(item["question"])[:self.max_q_len]
        para_offset = len(q_toks) + 2 # cls and sep
        item["wp_tokens"] = context_ann["all_doc_tokens"]  # [subword tokens]
        assert item["wp_tokens"][0] == "yes" and item["wp_tokens"][1] == "no"
        item["para_offset"] = para_offset  # start of paras para_offset = yes, para_offset+1=no para_offset+2=unanswerable
        max_toks_for_doc = self.max_seq_len - para_offset - 1
        if len(item["wp_tokens"]) > max_toks_for_doc:
            item["wp_tokens"] = item["wp_tokens"][:max_toks_for_doc]
        #TJH: is_split_into_words doesnt work:    
        #item["encodings"] = self.tokenizer.encode_plus(q_toks, text_pair=item["wp_tokens"], max_length=self.max_seq_len, return_tensors="pt", is_split_into_words=True)
        input_ids = torch.tensor([[self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(q_toks) + [self.tokenizer.sep_token_id] + self.tokenizer.convert_tokens_to_ids(item["wp_tokens"]) + [self.tokenizer.sep_token_id]],
                                 dtype=torch.int64)
        attention_mask = torch.tensor([[1] * input_ids.shape[1]], dtype=torch.int64)
        token_type_ids = torch.tensor([[0] * (len(q_toks)+2) + [1] * (input_ids.shape[1]-(len(q_toks)+2))], dtype=torch.int64)
        item["encodings"] = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}
        item["paragraph_mask"] = torch.zeros(item["encodings"]["input_ids"].size()).view(-1)
        item["paragraph_mask"][para_offset:-1] = 1  #TJH set para toks -> 1
        
        
        if self.train:
        #TODO if neg sample: point to [unused0]
        #TODO if full pos sample: point to yes/no/ans span
        #TODO if partial pos sample & classification and/or comparison question: point to [unused0]
        #TODO if partial pos sample & bridge & answer in para: point to answer or [unused0]??? bridge = len(sample['bridge']) > 1
            if item["ans_covered"]:  #fever = refutes/supports (neis excluded). hover = not_supported/supported where not_supported can be refuted or nei
                if item["gold_answer"][0] in ["yes", "SUPPORTED", "SUPPORTS"]:
                    starts, ends= [para_offset], [para_offset]
                elif item["gold_answer"][0] in ["no", "REFUTES", "NOT_SUPPORTED"]:
                    starts, ends= [para_offset + 1], [para_offset + 1]
                else:
                    matched_spans = match_answer_span(context_ann["context"], item["gold_answer"], self.simple_tok)
                    ans_starts, ans_ends= [], []
                    for span in matched_spans:
                        char_starts = [i for i in range(len(context_ann["context"])) if context_ann["context"].startswith(span, i)]
                        if len(char_starts) > 0:
                            char_ends = [start + len(span) - 1 for start in char_starts]
                            answer = {"text": span, "char_spans": list(zip(char_starts, char_ends))}
                            ans_spans = find_ans_span_with_char_offsets(answer, 
                                                                        context_ann["char_to_word_offset"], 
                                                                        context_ann["doc_tokens"], 
                                                                        context_ann["all_doc_tokens"], 
                                                                        context_ann["orig_to_tok_index"], 
                                                                        self.tokenizer)
                            for s, e in ans_spans:  #TJH accurate into context_ann["all_doc_tokens"] / item["wp_tokens"]
                                ans_starts.append(s)
                                ans_ends.append(e)
                    starts, ends = [], []
                    for s, e in zip(ans_starts, ans_ends):
                        if s >= len(item["wp_tokens"]):
                            continue
                        else:
                            s = min(s, len(item["wp_tokens"]) - 1) + para_offset  #TJH accurate into item["encodings"]["input_ids"][0]
                            e = min(e, len(item["wp_tokens"]) - 1) + para_offset
                            starts.append(s)
                            ends.append(e)
                    if len(starts) == 0:  # answer not in para
                        starts, ends = [para_offset + 2], [para_offset + 2]     # was CE ignore_index = -1 now [unused0] aka insuff evidence=unanswerable
            else:
                starts, ends= [para_offset + 2], [para_offset + 2] # was [-1] now [unused0] aka insuff evidence=unanswerable
                        
            item["starts"] = torch.LongTensor(starts)
            item["ends"] = torch.LongTensor(ends)

            if item["label"]:
                assert len(item["sp_sent_labels"]) == len(item["context_processed"]["sent_starts"])
        else:
            #     # for answer extraction
            item["doc_tokens"] = context_ann["doc_tokens"]
            item["tok_to_orig_index"] = context_ann["tok_to_orig_index"]

        # filter sentence offsets exceeding max sequence length
        sent_labels, sent_offsets = [], []
        for idx, s in enumerate(item["context_processed"]["sent_starts"]):
            if s >= len(item["wp_tokens"]):
                break
            if "sp_sent_labels" in item:
                sent_labels.append(item["sp_sent_labels"][idx])
            sent_offsets.append(s + para_offset)
            assert item["encodings"]["input_ids"].view(-1)[s+para_offset] == self.tokenizer.convert_tokens_to_ids("[unused1]")

        # supporting fact label
        item["sent_offsets"] = sent_offsets
        item["sent_offsets"] = torch.LongTensor(item["sent_offsets"])
        if self.train:
            item["sent_labels"] = sent_labels if len(sent_labels) != 0 else [0] * len(sent_offsets)
            item["sent_labels"] = torch.LongTensor(item["sent_labels"])
            item["ans_covered"] = torch.LongTensor([item["ans_covered"]])

        item["label"] = torch.LongTensor([item["label"]])
        return item



class AlternateSampler(Sampler):
    """
    Shuffle pairs of idxs assuming that even idx=pos example, odd=corresponding neg example
    Each pos/neg will tend to be on same gpu so shared normalisation.. 
    """
    def __init__(self, dset):
        self.num_samples = len(dset)
        self.idx_pairs = [(i, i+1) for i in range(0, self.num_samples, 2)]

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        indices = []
        random.shuffle(self.idx_pairs)
        for idxs in self.idx_pairs:
            for i in idxs:
                indices.append(i)
        return iter(indices)



class MhopSampler(Sampler):
    """
    Shuffle QA pairs not context, make sure data within the batch are from the same QA pair
    """

    def __init__(self, data_source, num_neg=9, n_gpu=8):
        # for each QA pair, sample negative paragraphs
        self.qid2gold = data_source.qid2gold
        self.qid2neg = data_source.qid2neg
        self.neg_num = num_neg
        self.n_gpu = n_gpu
        self.all_qids = list(self.qid2gold.keys())
        assert len(self.qid2gold) == len(self.qid2neg)

        self.q_num_per_epoch = len(self.qid2gold) - len(self.qid2gold) % self.n_gpu
        self._num_samples = self.q_num_per_epoch * (self.neg_num + 1)

    def __len__(self):
        return self._num_samples

    def __iter__(self):
        sample_indice = []
        random.shuffle(self.all_qids)
        
        # when use shared-normalization, passages for each question should be on the same GPU
        qids_to_use = self.all_qids[:self.q_num_per_epoch]
        for qid in qids_to_use:
            neg_samples = self.qid2neg[qid]
            random.shuffle(neg_samples)
            sample_indice += self.qid2gold[qid]
            sample_indice += neg_samples[:self.neg_num]
        return iter(sample_indice)

def qa_collate(samples, pad_id=0):
    if len(samples) == 0:
        return {}

    batch = {
        'input_ids': collate_tokens([s["encodings"]['input_ids'] for s in samples], pad_id),
        'attention_mask': collate_tokens([s["encodings"]['attention_mask'] for s in samples], 0),
        'paragraph_mask': collate_tokens([s['paragraph_mask'] for s in samples], 0),
        'label': collate_tokens([s["label"] for s in samples], -1),
        "sent_offsets": collate_tokens([s["sent_offsets"] for s in samples], 0),
        }

    # training labels
    if "starts" in samples[0]:
        batch["starts"] = collate_tokens([s['starts'] for s in samples], -1)
        batch["ends"] = collate_tokens([s['ends'] for s in samples], -1)
        # batch["ans_types"] = collate_tokens([s['ans_type'] for s in samples], -1)
        batch["sent_labels"] = collate_tokens([s['sent_labels'] for s in samples], 0)
        #batch["ans_covered"] = collate_tokens([s['ans_covered'] for s in samples], 0)

    # roberta does not use token_type_ids
    if "token_type_ids" in samples[0]["encodings"]:
        batch["token_type_ids"] = collate_tokens([s["encodings"]['token_type_ids']for s in samples], 0)

    batched = {
        "qids": [s["qid"] for s in samples],
        "passages": [s["passages"] for s in samples],
        "gold_answer": [s["gold_answer"] for s in samples],
        "sp_gold": [s["sp_gold"] for s in samples],
        "para_offsets": [s["para_offset"] for s in samples],
        "net_inputs": batch,
    }

    # for answer extraction
    if "doc_tokens" in samples[0]:
        batched["doc_tokens"] = [s["doc_tokens"] for s in samples]
        batched["tok_to_orig_index"] = [s["tok_to_orig_index"] for s in samples]
        batched["wp_tokens"] = [s["wp_tokens"] for s in samples]

    return batched
