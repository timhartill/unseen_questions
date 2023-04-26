"""
Dataset for stage 1 reranker and stage 2 evidence set scorer

Portions adapted from https://github.com/facebookresearch/multihop_dense_retrieval 
Two stage approach inspired by https://github.com/stanford-futuredata/Baleen

@author Tim Hartill

"""

import collections
import json
import random
import copy
import numpy as np

import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from mdr_basic_tokenizer_and_utils import SimpleTokenizer, para_has_answer, match_answer_span, find_ans_span_with_char_offsets
from text_processing import is_whitespace, get_sentence_list, split_into_sentences, create_sentence_spans

from utils import collate_tokens, get_para_idxs, consistent_bridge_format, encode_query_paras, encode_title_sents, context_toks_to_ids, flatten


def encode_query(sample, tokenizer, train, max_q_len, index, stage=1):
    """ Encode query as: question [unused2] title 1 | sent 1. sent 2 [unused2] title 2 | sent 0 ...
    sample = the (positive or corresponding negative) sample
    index  = the true index which alternates pos/negs. 
    train = whether to randomize para order (train) or keep fixed (dev)
    
    returns: 
        query: actually just the portion of the query following the question (not tokenised). '' in stage 2
        tokenised question: just the question part of the query (tokenised)
        idx of para to rerank (-1 to choose neg para) for stage 1 or in stage 2 -1 for neg example, -2 for pos
        build_to_hop: number of hops the query + next para cover eg 1=q_only->sp1, 2=q+sp1->sp2 (stage 1) IGNORED in stage 2
                      if build_to_hop=num_hops then query + next para is fully evidential (stage 1) 
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
        build_to_hop = sample['num_hops']  # comparatively few 3+ hops so always use full seq. Note there a a tiny number of fever examples > 4 hops

    para_list = []
    para_idxs = get_para_idxs(sample["pos_paras"])  # dict of {title: {idx: [idx in pos_paras]}} idx is a list to allow for possibility of same title duplicated in pos paras either for same title/difft paras or in case of FEVER same title, same para but difft sentence annotation
    sample['para_idxs'] = para_idxs  #save recalculating these in encode_context_stage2
    for step_paras_list in sample["bridge"]:
        if train and encode_pos:  # don't shuffle if a neg; use order last used for corresponding positive
            random.shuffle(step_paras_list)
        for title in step_paras_list:
            para_list.append( sample["pos_paras"][ para_idxs[title][0] ] )

    question = sample['question']  # + ' [unused2] '
    q_toks = tokenizer.tokenize(question)[:max_q_len]
    query = ''
    if stage == 1:
        for i, para in enumerate(para_list):
            if i+1 >= build_to_hop:
                if encode_pos:
                    rerank_para = para_idxs[ para['title'] ][0]  # always select next pos para
                else:
                    rerank_para = -1  # select either 1st (eval) or random (train) neg para when encoding context
                break
            query += ' [unused2] ' + encode_query_paras(para['text'], para['title'], 
                                                        para['sentence_spans'], para['sentence_labels'],
                                                        use_sentences=True, prepend_title=True, title_sep=' |')
    else:   # stage 2
        if encode_pos:
            rerank_para = -2
        else:
            rerank_para = -1
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
    (doc_tokens, char_to_word_offset, orig_to_tok_index, tok_to_orig_index, 
     all_doc_tokens, sent_starts) = context_toks_to_ids(context, tokenizer, sent_marker='[unused1]', special_toks=special_toks)
    
    sample["context_processed"] = {
            "doc_tokens": doc_tokens,                     # [whole words]
            "char_to_word_offset": char_to_word_offset,   # [char idx -> whole word idx]
            "orig_to_tok_index": orig_to_tok_index,       # [whole word idx -> subword idx]
            "tok_to_orig_index": tok_to_orig_index,       # [ subword token idx -> whole word token idx]
            "all_doc_tokens": all_doc_tokens,             # [ sub word tokens ]
            "context": context,                           # full context string including 'sentences' part of query
            "sent_starts": sent_starts,                   # [sentence start idx -> subword token idx]
            "sent_labels": s_labels,                      # [multihot sentence labels]
            "passage": para,                              # the pos or neg para {'title':.. 'text':..., pos/neg specific keys}
    }
    return sample


def encode_context_stage2(sample, tokenizer, rerank_para, train, special_toks=["[SEP]", "[unused0]", "[unused1]"]):
    """
    encode context for stage 2: add non-extractive answer choices, add sentence start markers [unused1] for sentence identification
    encode as: " [SEP] yes no [unused0] [SEP] [unused1] title2 | sent0 [unused1] title2 | sent2 [unused1] title0 | sent2 ..."
    if rerank_para == -1 then substitute some/all pos sents with neg sents -> label = insuff evidence
    else pos sample -> all pos sents present -> label = fully evidential
    In both pos/neg samples, additional neg sents are added
    if train negs to add are random
    if eval neg ordering is deterministic
    """
    para_titles = flatten(sample['bridge'])
    all_pos_sents = []
    all_neg_sents = []
    for t in para_titles:  # build list of all gold sentences plus separate list of all neg sentences from positive paras
        para = sample['pos_paras'][ sample['para_idxs'][t][0] ]
        pos_sents = encode_title_sents(para['text'], t.strip(), para['sentence_spans'], para['sentence_labels'])
        all_pos_sents.extend( pos_sents )
        neg_sent_idxs = []
        for i in range(len(para['sentence_spans'])): # Add neg sents from pos paras
            if i not in para['sentence_labels']:
                neg_sent_idxs.append(i)
        neg_sents = encode_title_sents(para['text'], t, para['sentence_spans'], neg_sent_idxs)
        all_neg_sents.extend( neg_sents )
        
    num_pos_initial = len(all_pos_sents) # annotation errors mean occasionally there are 0 positive sents
    all_pos_labels = [1] * num_pos_initial
    
    first_time = True
    while first_time or len(all_neg_sents) < num_pos_initial:
        first_time = False
        for i in range(2):  # add neg sents from neg paras to neg sentences list
            if train:
                para = random.choice(sample["neg_paras"])
            else: 
                para = sample["neg_paras"][i] # make eval deterministic
            t = para["title"].strip()
            sent_spans = create_sentence_spans( split_into_sentences(para["text"]) )
            neg_sent_idxs = list(range(len(sent_spans)))
            neg_sents = encode_title_sents(para['text'], t, sent_spans, neg_sent_idxs)
            all_neg_sents.extend( neg_sents )
    if train:
        random.shuffle(all_neg_sents)
    
    if rerank_para == -1: # neg sample - replace some/all pos sents with neg sents, CLS label will be insuff evidence
        curr_pos_idxs = list(range(num_pos_initial))
        if train:
            divisor = random.choice([2,3])
            random.shuffle(curr_pos_idxs)
        else:
            divisor = 2
        if sample['src'] not in ['fever', 'scifact']:  # scifact sent evidentiality is "or" OR "and" so treat like "fever or"
            firstnegidx = num_pos_initial // divisor
        else:
            firstnegidx = 0  # fever sent evidentiality is "or" not "and" so set all neg sample sents to neg_sents since partially replacing pos sents doesnt work wrt label
        neg_idx = -1
        for i in range(firstnegidx, num_pos_initial):
            neg_idx += 1
            all_pos_sents[i] = all_neg_sents[neg_idx]
            all_pos_labels[i] = 0
        all_neg_sents = all_neg_sents[neg_idx+1:]
    
    max_sents = random.choice([7,8,9]) if train else 9
    num_to_add = max_sents - num_pos_initial
    if num_to_add > 0:
        all_pos_sents.extend( all_neg_sents[:num_to_add] )  # additional negs
        num_to_add = len(all_pos_sents) - len(all_pos_labels)
        all_pos_labels += [0] * num_to_add
        
    #shuffle sents preserving label mapping
    all_pair = list(zip(all_pos_sents, all_pos_labels))
    if train:
        random.shuffle(all_pair)
    context = ' '.join([s[0] for s in all_pair])
    s_labels = [s[1] for s in all_pair]
    pos_sent_idxs = [i for i,s in enumerate(s_labels) if s == 1]  

    context = " [SEP] yes no [unused0] [SEP] " + context  # ELECTRA tokenises yes, no to single tokens
    (doc_tokens, char_to_word_offset, orig_to_tok_index, tok_to_orig_index, 
     all_doc_tokens, sent_starts) = context_toks_to_ids(context, tokenizer, sent_marker='[unused1]', special_toks=special_toks)
    
    sample["context_processed"] = {
            "doc_tokens": doc_tokens,                     # [whole words]
            "char_to_word_offset": char_to_word_offset,   # [char idx -> whole word idx]
            "orig_to_tok_index": orig_to_tok_index,       # [whole word idx -> subword idx]
            "tok_to_orig_index": tok_to_orig_index,       # [ subword token idx -> whole word token idx]
            "all_doc_tokens": all_doc_tokens,             # [ sub word tokens ]
            "context": context,                           # full context string ie everything after question
            "sent_starts": sent_starts,                   # [sentence start idx -> subword token idx]
            "sent_labels": s_labels,                      # [multihot sentence labels]
            "passage": {'pos_sent_idxs': pos_sent_idxs},  # dict for stage1 format compatability: was the pos or neg para {'title':.. 'text':..., pos/neg specific keys}
    }
    return sample




class Stage1Dataset(Dataset):

    def __init__(self, args, tokenizer, data_path, train=False):
        """ Set up dataset as alternating pos-corresponding neg samples. Neg samples have insuff evidence label
        """
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
            if sample['answers'][0] in ["SUPPORTED", "SUPPORTS"]: #fever = refutes/supports (neis excluded). hover = not_supported/supported where not_supported can be refuted or nei
                sample['answers'][0] = 'yes'
            elif sample['answers'][0] in ["REFUTES", "NOT_SUPPORTED"]:
                sample['answers'][0] = 'no'
            data.append(sample)                 # pos example - para is always pos but may not be final
            neg_sample = copy.deepcopy(sample)
            neg_sample['_id'] += '__neg__'
            neg_sample['answers'] = ['[unused0]']   # neg sample always has 'insufficient evidence' answer
            neg_sample['sp_gold_single'] = [[]]     # neg sample 'correct' title/sents dont exist 
            data.append(neg_sample)  # neg example - para is always neg but may not be final
        self.data = data
        print(f"Data size {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        #query, q_toks, rerank_para, build_to_hop = encode_query_stage1(sample, tokenizer, train, max_q_len, index)
        query, q_toks, rerank_para, build_to_hop = encode_query(sample, self.tokenizer, self.train, self.max_q_len, index)
        if index % 2 == 0:                                                  # encoding positive sample -> make neg query be encoded the same way
            self.data[index+1]['last_build_to_hop'] = build_to_hop          #force corresponding neg to build to same # of hops as the positive
            self.data[index+1]['bridge'] = copy.deepcopy(sample['bridge'])  # force neg to use same para order as positive
        #item = encode_context_stage1(sample, tokenizer, rerank_para, train, query)
        item = encode_context_stage1(sample, self.tokenizer, rerank_para, self.train, query)
        item["index"] = index
        context_ann = item["context_processed"]
        #q_toks = self.tokenizer.tokenize(item["question"])[:self.max_q_len]
        para_offset = len(q_toks) + 1 #  cls 
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
        item["paragraph_mask"][para_offset:-1] = 1  #set sentences part of query + para toks -> 1
        #NOTE if query very long then 1st [SEP] will be the EOS token at pos 511 & ans_offset will be at 512 over max seq len...
        #ans_offset = torch.where(input_ids[0] == tokenizer.sep_token_id)[0][0].item()+1
        ans_offset = torch.where(input_ids[0] == self.tokenizer.sep_token_id)[0][0].item()+1  # tok after 1st [SEP] = yes = start of non extractive answer options
        if ans_offset >= 509: #non extractive ans options + eval para truncated due to very long query
            ans_offset = -1
        item['insuff_offset'] = torch.LongTensor([ans_offset+2])   # idx of insuff token  if no insuff token = 1

        if self.train:
            #if neg sample: point to [unused0]/insufficient evidence
            #if full pos sample: point to yes/no/ans span
            #if partial pos sample: point to [unused0]
            if ans_offset == -1:
                starts, ends = [-1], [-1]  #CE will ignore -1
            elif rerank_para > -1 and build_to_hop >= item['num_hops']: # if pos & fully evidential ie query + next para = full para set
                if item["answers"][0] in ["yes", "SUPPORTED", "SUPPORTS"]: # ans mapped to y/n in init above but kept here in case want to change back
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
                        if s >= len(item["wp_tokens"]) or s < 0: 
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

        else:   # eval sample 
            item["full"] = torch.LongTensor([1 if build_to_hop >= item['num_hops'] else 0])
            item["act_hops"] = torch.LongTensor([build_to_hop])
            item["doc_tokens"] = context_ann["doc_tokens"]
            item["tok_to_orig_index"] = context_ann["tok_to_orig_index"]
            if rerank_para > -1 and build_to_hop < item['num_hops']: # update labels if pos but partial sample
                item = copy.deepcopy(item) # need to override answer
                item['answers'] = ['[unused0]']
        
        if rerank_para > -1: # negs already have sp_gold_single = [[]]
            sp_gold_single = []
            for sentence_label in item["context_processed"]["passage"]["sentence_labels"]:
                if sentence_label < len(item["context_processed"]["passage"]["sentence_spans"]):
                    sp_gold_single.append( [item["context_processed"]["passage"]["title"], sentence_label] )  # [ [title, sidx1], [title, sidx2], ... ]
            if sp_gold_single == []:
                sp_gold_single = [[]]
            item['sp_gold_single'] = sp_gold_single
        

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


class Stage2Dataset(Dataset):

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
            if sample['answers'][0] in ["SUPPORTED", "SUPPORTS"]: #fever = refutes/supports (neis excluded). hover = not_supported/supported where not_supported can be refuted or nei
                sample['answers'][0] = 'yes'
            elif sample['answers'][0] in ["REFUTES", "NOT_SUPPORTED"]:
                sample['answers'][0] = 'no'
            data.append(sample)                      # pos example - para is always pos but may not be final
            neg_sample = copy.deepcopy(sample)
            neg_sample['_id'] += '__neg__'
            neg_sample['answers'] = ['[unused0]']    # neg sample always has 'insufficient evidence' answer
            #neg_sample['sp_gold_single'] = []       # neg sample but 'correct' title/sents may exist amongst the neg sents 
            data.append(neg_sample)                  # neg example - para/sentences is always neg but may not be final
        self.data = data
        print(f"Data size {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        #query, q_toks, rerank_para, build_to_hop = encode_query(sample, tokenizer, train, max_q_len, index, stage=2)
        query, q_toks, rerank_para, build_to_hop = encode_query(sample, self.tokenizer, self.train, self.max_q_len, index, stage=2)
        if index % 2 == 0:                                                  # encoding positive sample -> make neg query be encoded the same way
            self.data[index+1]['last_build_to_hop'] = build_to_hop          # force corresponding neg to build to same # of hops as the positive (ignored in stage 2)
            self.data[index+1]['bridge'] = copy.deepcopy(sample['bridge'])  # force neg to use same para order as positive
        #item = encode_context_stage2(sample, tokenizer, rerank_para, train)
        item = encode_context_stage2(sample, self.tokenizer, rerank_para, self.train)
        item["index"] = index
        context_ann = item["context_processed"]
        #q_toks = self.tokenizer.tokenize(item["question"])[:self.max_q_len]
        para_offset = len(q_toks) + 1 #  cls 
        item["wp_tokens"] = context_ann["all_doc_tokens"]  # [subword tokens]
        #assert item["wp_tokens"][0] == "yes" and item["wp_tokens"][1] == "no"
        item["para_offset"] = para_offset  # 1st tok after basic question ie start of sentences component of query in stage 1 or start of context in stage 2
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
        item["paragraph_mask"][para_offset:-1] = 1  #set sentences part of query + para toks -> 1
        #NOTE if query very long then 1st [SEP] will be the EOS token at pos 511 & ans_offset will be at 512 over max seq len...
        #ans_offset = torch.where(input_ids[0] == tokenizer.sep_token_id)[0][0].item()+1
        ans_offset = torch.where(input_ids[0] == self.tokenizer.sep_token_id)[0][0].item()+1  # tok after 1st [SEP] = yes = start of non extractive answer options
        if ans_offset >= 509: #non extractive ans options + eval para truncated due to very long query
            ans_offset = -1  
        item['insuff_offset'] = torch.LongTensor([ans_offset+2])   # idx of insuff token  if no insuff token = 1

        if self.train:
            #if neg sample: point to [unused0]/insufficient evidence
            #if full pos sample: point to yes/no/ans span
            #unlike stage 1 there are no partial pos samples but sent annot errs mean there are some "pos" samples that have no valid sent labels..
            if ans_offset == -1:
                starts, ends = [-1], [-1]  #CE will ignore -1
            elif rerank_para != -1 and item["context_processed"]["passage"]["pos_sent_idxs"] != []: # if pos ie fully evidential ie all positive sents present (plus a few negs as distractors)
                if item["answers"][0] in ["yes", "SUPPORTED", "SUPPORTS"]: # ans mapped to y/n in init above but kept here in case want to change back
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
                        if s >= len(item["wp_tokens"]) or s < 0: 
                            continue
                        else:
                            s = min(s, len(item["wp_tokens"]) - 1) + para_offset  #TJH accurate into item["encodings"]["input_ids"][0]
                            e = min(e, len(item["wp_tokens"]) - 1) + para_offset
                            starts.append(s)
                            ends.append(e)
                    if len(starts) == 0:  # answer not in para
                        starts, ends = [ans_offset + 2], [ans_offset + 2]     # was CE ignore_index = -1 now [unused0] aka insuff evidence=unanswerable
            else:  # neg -> not all pos sents present -> insuff evidence to answer
                starts, ends= [ans_offset + 2], [ans_offset + 2] # was [-1] now [unused0] aka insuff evidence=unanswerable
                        
            item["starts"] = torch.LongTensor(starts)
            item["ends"] = torch.LongTensor(ends)

        else:   # eval sample 
            item["full"] = torch.LongTensor([1 if build_to_hop >= item['num_hops'] else 0]) # ignored in stage 2, just for format compatability with stage1
            item["act_hops"] = torch.LongTensor([build_to_hop])                             # ignored in stage 2, just for format compatability with stage1
            item["doc_tokens"] = context_ann["doc_tokens"]
            item["tok_to_orig_index"] = context_ann["tok_to_orig_index"]
        
        # negs can have (partial) pos sents. if no pos sents, sp_gold_single = [] (note stage 1 neg label is [[]] )
        item['sp_gold_single'] = item["context_processed"]["passage"]["pos_sent_idxs"]

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
        if rerank_para == -1 or item['sp_gold_single'] == []:  # if neg or "was supposed to be pos but sent annot errors mean no pos sents"
            item["label"] = torch.LongTensor([0])  # sents not fully evidential
        else:  # pos
            item["label"] = torch.LongTensor([1]) # sents are fully evidential
        return item


class AlternateSampler(Sampler):
    """
    Shuffle pairs of idxs assuming that even idx=pos example, odd=corresponding neg example
    Each pos/neg will be on same gpu since training on 1 gpu so shared normalisation.. 
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


def stage_collate(samples, pad_id=0):
    if len(samples) == 0:
        return {}
    
    batch = {
        'input_ids': collate_tokens([s["encodings"]['input_ids'] for s in samples], pad_id),
        'attention_mask': collate_tokens([s["encodings"]['attention_mask'] for s in samples], 0),
        'paragraph_mask': collate_tokens([s['paragraph_mask'] for s in samples], 0),
        'label': collate_tokens([s["label"] for s in samples], -1),
        "sent_offsets": collate_tokens([s["sent_offsets"] for s in samples], 0),
        "sent_labels": collate_tokens([s['sent_labels'] for s in samples], 0),
        "insuff_offset": collate_tokens([s['insuff_offset'] for s in samples], 1)  # make padding val same as no insuff tok value ie 1
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
        "sp_gold": [s["sp_gold_single"] for s in samples],  # override full sp_gold with the subset for the para relevant to sample query
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
        batched["act_hops"] = [s["act_hops"] for s in samples]
        batched["question"] = [s["question"] for s in samples]
        batched["context"] = [s["context_processed"]["context"] for s in samples]

    return batched
