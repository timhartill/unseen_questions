"""
Dataset for stage 1 extractor

encode_context_stage1() adapted from prepare() fn in https://github.com/facebookresearch/multihop_dense_retrieval 
encode_query_stage1() inspired by https://github.com/stanford-futuredata/Baleen

@author Tim Hartill

"""

import collections
import json
import random
import copy
#import numpy as np

import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from mdr_basic_tokenizer_and_utils import SimpleTokenizer, para_has_answer, match_answer_span, find_ans_span_with_char_offsets
from text_processing import is_whitespace, get_sentence_list, split_into_sentences

from utils import collate_tokens, get_para_idxs, consistent_bridge_format, encode_query_paras


def encode_query_stage1(sample, tokenizer, train, max_q_len, index):
    """ Encode query as: question [unused2] title 1 | sent 1. sent 2 [unused2] title 2 | sent 0 ...
    sample = the (positive or corresponding negative) sample
    index  = the true index which alternates pos/negs. 
    train = whether to randomize para order (train) or keep fixed (dev)
    
    returns: 
        query: actually just the portion of the query following the question (not tokenised)
        tokenised question: just the question part of the query (tokenised) 
        idx of para to rerank (-1 to choose neg para)
        build_to_hop: number of hops the query + next para cover eg 1=q_only->sp1, 2=q+sp1->sp2  if build_to_hop=num_hops then query + next para is fully evidential
    """
    if index % 2 == 0:
        encode_pos = True
    else:
        encode_pos = False
               
    if sample['num_hops'] == 1:
        build_to_hop = 1
    elif sample['num_hops'] == 2:
        if train:
            if encode_pos or sample.get('last_build_to_hop') is None:
                build_to_hop = random.randint(1,2)
                sample['last_build_to_hop'] = build_to_hop
            else:
                build_to_hop = sample['last_build_to_hop']
        else: # make deterministic for eval half taking 1st hop. half second hop
            orig_index = index // 2
            if orig_index % 2 == 0:
                build_to_hop = 2
            else:
                build_to_hop = 1
    else:
        build_to_hop = sample['num_hops']  # comparatively few 3+ hops so always use full seq. Note there a a tiny number of fevr examples > 4 hops

    para_list = []
    para_idxs = get_para_idxs(sample["pos_paras"])
    for step_paras_list in sample["bridge"]:
        if train and encode_pos:  # don't shuffle if a neg; use order last used for corresponding positive
            random.shuffle(step_paras_list)
        for title in step_paras_list:
            para_list.append( sample["pos_paras"][ para_idxs[title][0] ] )

    question = sample['question']  # + ' [unused2] '
    q_toks = tokenizer.tokenize(question)[:max_q_len]
    query = ''
    for i, para in enumerate(para_list):
        if i+1 >= build_to_hop:
            if encode_pos:
                rerank_para = para_idxs[para['title']][0]
            else:
                rerank_para = -1
            break
        query += ' [unused2] ' + encode_query_paras(para['text'], para['title'], 
                                                    para['sentence_spans'], para['sentence_labels'],
                                                    use_sentences=True, prepend_title=True, title_sep=' |')
    #q_toks = tokenizer.tokenize(query)[:max_q_len]
    #print(f"enc_query. index:{index} encode_pos:{encode_pos} rerank_para:{rerank_para}  build_to_hop:{build_to_hop}")

    return query, q_toks, rerank_para, build_to_hop



def encode_context_stage1(sample, tokenizer, rerank_para, train, query, special_toks=["[SEP]", "[unused0]", "[unused1]"]):
    """
    encode context: add post question part of query, non-extractive answer choices, add sentence start markers [unused1] for sentence identification
    encode as: "sentences part of query [SEP] yes no [unused0] [SEP] title [unused1] sent0 [unused1] sent1 [unused1] sent2 ..."
    if rerank_para = -1 then add a neg para from neg_paras key: random if train, first neg if eval
    """
    def _process_pos(para):
        """ positive para """
        title, sentence_spans, sentence_labels = para["title"].strip(), para["sentence_spans"], set(para["sentence_labels"])
        sents = get_sentence_list(para["text"], sentence_spans)
        pre_sents = []
        s_labels = []
        for idx, sent in enumerate(sents):
            pre_sents.append("[unused1] " + sent.strip())
            s_labels.append( 1 if idx in sentence_labels else 0 )
        return title + " " + " ".join(pre_sents), s_labels
    
    def _process_neg(para):
        """ neg para """
        title = para["title"].strip()
        sents = split_into_sentences(para["text"])
        pre_sents = []
        s_labels = []
        for idx, sent in enumerate(sents):
            pre_sents.append("[unused1] " + sent.strip())
            s_labels.append(0)
        return title + " " + " ".join(pre_sents), s_labels
          
   
    if rerank_para > -1:
        para = sample["pos_paras"][rerank_para]
        context, s_labels = _process_pos(para)
    else:
        if train:
            neg_para = random.choice(sample["neg_paras"])
        else:
            neg_para = sample["neg_paras"][0] # make eval deterministic
        context, s_labels = _process_neg(neg_para)
        para = neg_para

    context = query + " [SEP] yes no [unused0] [SEP] " + context  # ELECTRA tokenises yes, no to single tokens
    doc_tokens = []  # ['word1', 'word2', ..]
    char_to_word_offset = []  # list with each char -> idx into doc_tokens
    prev_is_whitespace = True
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

    sample["context_processed"] = {
        "doc_tokens": doc_tokens,                     # [whole words]
        "char_to_word_offset": char_to_word_offset,   # [char idx -> whole word idx]
        "orig_to_tok_index": orig_to_tok_index,       # [whole word idx -> subword idx]
        "tok_to_orig_index": tok_to_orig_index,       # [ subword token idx -> whole word token idx]
        "all_doc_tokens": all_doc_tokens,             # [ sub word tokens ]
        "context": context,                           # full context string
        "sent_starts": sent_starts,                   # [sentence start idx -> subword token idx]
        "sent_labels": s_labels,                      # [multihot sentence labels]
        "passage": para,                              # the pos or neg para {'title':.. 'text':..., pos/neg specific keys}
    }
    return sample



class Stage1Dataset(Dataset):

    def __init__(self, args, tokenizer, data_path, train=False):
        self.data_path = data_path
        print(f"Train:{train} Loading from: {data_path}..")
        samples = [json.loads(l) for l in tqdm(open(data_path).readlines())]
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_c_len
        self.max_q_len = args.max_q_len
        self.train = train
        self.simple_tok = SimpleTokenizer()
        data = []  # Each alternate sample will be a blank placeholder denoting a negative for preceding positive
        print("Standardizing formats...")
        for sample in tqdm(samples):
            if sample["question"].endswith("?"):
                sample["question"] = sample["question"][:-1]
            consistent_bridge_format(sample)
            data.append(sample)
            data.append(copy.deepcopy(sample))  # dummy entry for neg example - construct actual neg from data[index-1] 
        self.data = data
        print(f"Data size {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        #query, q_toks, rerank_para, build_to_hop = encode_query_stage1(sample, tokenizer, train, max_q_len, index)
        query, q_toks, rerank_para, build_to_hop = encode_query_stage1(sample, self.tokenizer, self.train, self.max_q_len, index)
        if index % 2 == 0:  # encoding positive sample
            self.data[index+1]['last_build_to_hop'] = build_to_hop #force corresponding neg to build to same # of hops as the positive
            self.data[index+1]['bridge'] = copy.deepcopy(sample['bridge']) # force neg to use same para order as positive
        #item = encode_context_stage1(sample, tokenizer, rerank_para, train, query)
        item = encode_context_stage1(sample, self.tokenizer, rerank_para, self.train, query)
        item["index"] = index
        context_ann = item["context_processed"]
        #q_toks = self.tokenizer.tokenize(item["question"])[:self.max_q_len]
        para_offset = len(q_toks) + 1 #  + cls 
        item["wp_tokens"] = context_ann["all_doc_tokens"]  # [subword tokens]
        #assert item["wp_tokens"][0] == "yes" and item["wp_tokens"][1] == "no"
        item["para_offset"] = para_offset  # 1st tok after basic question ie start of sentences component of query
        max_toks_for_doc = self.max_seq_len - para_offset - 1
        if len(item["wp_tokens"]) > max_toks_for_doc:
            item["wp_tokens"] = item["wp_tokens"][:max_toks_for_doc]
        #input_ids = torch.tensor([[tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(q_toks) + tokenizer.convert_tokens_to_ids(item["wp_tokens"]) + [tokenizer.sep_token_id]], dtype=torch.int64)
        input_ids = torch.tensor([[self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(q_toks) + self.tokenizer.convert_tokens_to_ids(item["wp_tokens"]) + [self.tokenizer.sep_token_id]],
                                 dtype=torch.int64)
        attention_mask = torch.tensor([[1] * input_ids.shape[1]], dtype=torch.int64)
        token_type_ids = torch.tensor([[0] * para_offset + [1] * (input_ids.shape[1]-para_offset)], dtype=torch.int64)
        item["encodings"] = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}
        item["paragraph_mask"] = torch.zeros(item["encodings"]["input_ids"].size()).view(-1)
        item["paragraph_mask"][para_offset:-1] = 1  #TJH set para toks -> 1
        ans_offset = torch.where(input_ids[0] == self.tokenizer.sep_token_id)[0][0].item()+1  # tok after 1st [SEP] = yes = start of non extractive answer options
        
        if self.train:
            #if neg sample: point to [unused0]/insufficient evidence
            #if full pos sample: point to yes/no/ans span
            #if partial pos sample: point to [unused0]
            if rerank_para > -1 and build_to_hop >= item['num_hops']: # if pos & fully evidential ie query + next para = full para set
                if item["answers"][0] in ["yes", "SUPPORTED", "SUPPORTS"]: #fever = refutes/supports (neis excluded). hover = not_supported/supported where not_supported can be refuted or nei
                    starts, ends= [ans_offset], [ans_offset]
                elif item["answers"][0] in ["no", "REFUTES", "NOT_SUPPORTED"]:
                    starts, ends= [ans_offset + 1], [ans_offset + 1]
                else:
                    #matched_spans = match_answer_span(context_ann["context"], item["answers"], simple_tok)
                    matched_spans = match_answer_span(context_ann["context"], item["answers"], self.simple_tok)
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
                        starts, ends = [ans_offset + 2], [ans_offset + 2]     # was CE ignore_index = -1 now [unused0] aka insuff evidence=unanswerable
            else:
                starts, ends= [ans_offset + 2], [ans_offset + 2] # was [-1] now [unused0] aka insuff evidence=unanswerable
                        
            item["starts"] = torch.LongTensor(starts)
            item["ends"] = torch.LongTensor(ends)

        else:   # for answer extraction
            item["full"] = torch.LongTensor([1 if build_to_hop >= item['num_hops'] else 0])
            item["doc_tokens"] = context_ann["doc_tokens"]
            item["tok_to_orig_index"] = context_ann["tok_to_orig_index"]

        # filter sentence offsets exceeding max sequence length
        sent_labels, sent_offsets = [], []
        for idx, s in enumerate(item["context_processed"]["sent_starts"]):
            if s >= len(item["wp_tokens"]): #if wp_tokens truncated, sent labels could be invalid
                break
            sent_labels.append(item["context_processed"]["sent_labels"][idx])
            sent_offsets.append(s + para_offset)
            assert item["encodings"]["input_ids"].view(-1)[s+para_offset] == 2  #self.tokenizer.convert_tokens_to_ids("[unused1]")

        item["sent_offsets"] = torch.LongTensor(sent_offsets)
        item["sent_labels"] = torch.LongTensor(sent_labels)
        item["label"] = torch.LongTensor([1  if rerank_para > -1 else 0]) # pos sample, next para always evidential, neg sample, next para never evidential 
        return item



class AlternateSampler(Sampler):
    """
    Shuffle pairs of idxs assuming that even idx=pos example, odd=corresponding neg example
    Each pos/neg will tend to be on same gpu so pseudo shared normalisation.. 
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
        yield from iter(indices)



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


def stage1_collate(samples, pad_id=0):
    if len(samples) == 0:
        return {}
    
    batch = {
        'input_ids': collate_tokens([s["encodings"]['input_ids'] for s in samples], pad_id),
        'attention_mask': collate_tokens([s["encodings"]['attention_mask'] for s in samples], 0),
        'paragraph_mask': collate_tokens([s['paragraph_mask'] for s in samples], 0),
        'label': collate_tokens([s["label"] for s in samples], -1),
        "sent_offsets": collate_tokens([s["sent_offsets"] for s in samples], 0),
        "sent_labels": collate_tokens([s['sent_labels'] for s in samples], 0)
        }

    # training labels
    if "starts" in samples[0]:
        batch["starts"] = collate_tokens([s['starts'] for s in samples], -1)
        batch["ends"] = collate_tokens([s['ends'] for s in samples], -1)
        
    # roberta does not use token_type_ids but electra does
    if "token_type_ids" in samples[0]["encodings"]:
        batch["token_type_ids"] = collate_tokens([s["encodings"]['token_type_ids']for s in samples], 0)

    batched = {
        "qids": [s["_id"] for s in samples],
        "passages": [[s["context_processed"]['passage']] for s in samples],
        "gold_answer": [s["answers"] for s in samples],
        "sp_gold": [s["sp_gold"] for s in samples],
        "para_offsets": [s["para_offset"] for s in samples],
        "net_inputs": batch,
    }
    
    if "index" in samples[0]:
        batched["index"] = [s["index"] for s in samples]

    # for answer extraction
    if "doc_tokens" in samples[0]:  # only for eval
        batched["doc_tokens"] = [s["doc_tokens"] for s in samples]
        batched["tok_to_orig_index"] = [s["tok_to_orig_index"] for s in samples]
        batched["wp_tokens"] = [s["wp_tokens"] for s in samples]
        batched["full"] = [s["full"] for s in samples]

    return batched
