"""
Model and Dataset for rr model training.


@author Tim Hartill

"""

import json
import random
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from transformers import AutoModel

from utils import collate_tokens, move_to_cuda
from text_processing import split_into_sentences

NON_EXTRACTIVE_OPTIONS = ' [SEP] yes no [unused0] [SEP] '  # for s1 model..
GENERIC_TITLE = 'Explanation'  # s1/s2 models expects a title


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class RRModel(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        if args.__dict__.get('model_name_stage') is not None:
            self.model_name = args.model_name_stage
        else:    
            self.model_name = args.model_name
        self.debug = args.debug
        self.debug_count = 3
        self.encoder = AutoModel.from_pretrained(self.model_name)
        if "electra" in self.model_name:
            self.pooler = BertPooler(config)
        self.rank = nn.Linear(config.hidden_size, 1) 

        
    def forward(self, batch):
        outputs = self.encoder(batch['input_ids'], batch['attention_mask'], batch.get('token_type_ids', None))
        if "electra" in self.model_name:
            sequence_output = outputs[0]  # [0]=raw seq output [bs, seq_len, hs]
            pooled_output = self.pooler(sequence_output) # [bs, hs]
        else:
            sequence_output, pooled_output = outputs[0], outputs[1]
        rank_score = self.rank(pooled_output)  # [bs, 1]
                
        if self.training:
            rank_target = batch["label"]
            rank_loss = F.binary_cross_entropy_with_logits(rank_score, rank_target.float(), reduction="sum")                           
            if self.debug and self.debug_count > 0:
                print(f"LOSSES: rank_loss:{rank_loss}")
                self.debug_count -= 1
            return rank_loss.unsqueeze(0)        
        return { 'rank_score': rank_score }      # [bs,1] is para evidential 0<->1


class RRDataset(Dataset):
    """ RR Model dataset for training"""
    def __init__(self, args, tokenizer, data_path, train=False):
        self.data_path = data_path
        print(f"Train:{train} Loading from: {data_path}..")
        samples = [json.loads(l) for l in tqdm(open(data_path).readlines())]
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_c_len # overall max seq len not max len of context portion..
        self.max_q_len = args.max_q_len
        self.mc_strip_prob = args.mc_strip_prob # prob of stripping mc options from query where these exist
        self.single_pos_samples = args.single_pos_samples
        self.train = train
        self.debug = args.debug
        self.debug_count = 3
        data = []  # Each alternate sample will be a placeholder denoting a negative for preceding positive
        print("Standardizing formats...")
        if self.train and args.single_pos_samples:
            print("Splitting train data into single-positive samples...")
            newsamples = []
            for sample in tqdm(samples):
                for i, pos in enumerate(sample['pos_paras']):
                    newsample = copy.deepcopy(sample)
                    newsample['pos_paras'] = [ pos ]
                    newsample['_id'] = sample['_id'] + '__' + str(i)
                    newsamples.append(newsample)
            samples = newsamples
        for sample in tqdm(samples):
            if sample["question"].endswith("?"):
                sample["question"] = sample["question"][:-1]
                
            # already preprocessed to yes/no but for consistency..
            if sample['answers'][0] in ["SUPPORTED", "SUPPORTS"]: #fever = refutes/supports (neis excluded). hover = not_supported/supported where not_supported can be refuted or nei
                sample['answers'][0] = 'yes'
            elif sample['answers'][0] in ["REFUTES", "NOT_SUPPORTED"]:
                sample['answers'][0] = 'no'
            sample['_id'] = sample['src'] + '___' + sample['_id']  # in the very unlikely case we get the same id from different datasets in one batch..
            sample['label'] = 1
            if sample.get('mc_options') is None:
                sample['mc_options'] = ''
            if sample.get('context') is None:
                sample['context'] = ''
            data.append(sample)                      # pos example - rationale to score will be drawn from pos_paras key
            neg_sample = copy.deepcopy(sample)
            neg_sample['_id'] += '__neg__'
            neg_sample['label'] = 0
            data.append(neg_sample)                  # neg example - rationale to score will be drawn from neg_paras key
        self.data = data
        print(f"Data size: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        # encode query [c+]q[+mc]
        q = sample['question'].strip()
        m = sample['mc_options'].strip()
        if m != '':
            if self.train and random.random() <= self.mc_strip_prob:
                m = ''
            else:
                m = ' ' + m
        q = q + m
        q_toks = self.tokenizer.tokenize(q)[:self.max_q_len]
        c = sample['context'][:600].strip()  # initial context. IIRC only. Not present in training...
        if c != '':
            if c[-1] != '.':
                c += '.'
            c += ' '
            
        para_offset = len(q_toks) + 1 #  cls
        q_ids = [self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(q_toks)
        max_toks_for_doc = self.max_seq_len - para_offset - 1
        if max_toks_for_doc <= 2 and self.debug and self.debug_count > 0:
            print(f"Query too long: _id:{sample['_id']}")
            self.debug_count -= 1
        
        # encode rationale
        if sample['label'] == 1:
            key = 'pos_paras'
        else:
            key = 'neg_paras'
        if self.train:
            para = random.choice(sample[key])
        else:
            para = sample[key][0]  # make dev deterministic. Note pos & neg paras were shuffled during preprocessing
        rat = " [SEP] " + c + para['text'].strip()
        r_toks = self.tokenizer.tokenize(rat)
        if len(r_toks) > max_toks_for_doc:
            if self.debug and self.debug_count > 0:
                if len(r_toks) > 511:
                    print(f"RAT > 511 toks: index:{index}")
                print(f"Rat truncated. index:{index} _id:{sample['_id']}")
                self.debug_count -= 1
            r_toks = r_toks[:max_toks_for_doc]
        r_ids = self.tokenizer.convert_tokens_to_ids(r_toks)

        input_ids = torch.tensor([q_ids + r_ids + [self.tokenizer.sep_token_id]], dtype=torch.int64)
        attention_mask = torch.tensor([[1] * input_ids.shape[1]], dtype=torch.int64)
        token_type_ids = torch.tensor([[0] * para_offset + [1] * (input_ids.shape[1]-para_offset)], dtype=torch.int64)
        item = {}
        item["encodings"] = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}
        item["label"] = torch.LongTensor([sample['label']])
        if not self.train:
            item["question"] = q
            item["context"] = para['text'].strip()
            item["answers"] = sample["answers"]
            item["index"] = index
            item["_id"] = sample["_id"]
            item["para_offset"] = para_offset  # 1st tok after query ie [SEP]

        return item


class RREvalDataset(Dataset):
    """ RR Model simplified Eval dataset
    input samples: list of dict_keys(['question', 'answer', 'q_only', 'mc_options', 'context', 'iter_context', 'iter_context_ev_score'])
    q_only key contains the base question minus mc options or other context
    """
    def __init__(self, args, tokenizer, samples, score_llm=True):
        self.score_llm = score_llm
        if score_llm:
            self.expl_key = 'context'       # llm-generated explanation/rationale/context including initial para for iirc
        else:
            self.expl_key = 'iter_context'  # iterator-generated context including init para for iirc            
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_c_len # overall max seq len not max len of context portion..
        self.max_q_len = args.max_q_len
        self.debug = args.debug
        self.debug_count = 3
        self.data = samples
        print(f"Data size: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        # encode query [c+]q[+mc]
        q = sample['q_only'].strip()
        m = sample['mc_options'].strip()
        if m != '':
            m = ' ' + m
        q = q + m
        q_toks = self.tokenizer.tokenize(q)[:self.max_q_len]
        c = sample[self.expl_key].strip()  # context to score
        if c != '':
            if c[-1] != '.':
                c += '.'

        para_offset = len(q_toks) + 1 #  cls
        q_ids = [self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(q_toks)
        max_toks_for_doc = self.max_seq_len - para_offset - 1
        if max_toks_for_doc <= 2 and self.debug and self.debug_count > 0:
            print(f"Query too long: _id:{sample.get('_id')}")
            self.debug_count -= 1
        
        rat = " [SEP] " + c
        r_toks = self.tokenizer.tokenize(rat)
        if len(r_toks) > max_toks_for_doc:
            if self.debug and self.debug_count > 0:
                if len(r_toks) > 511:
                    print(f"RAT > 511 toks: index:{index}")
                print(f"Rat truncated. index:{index} _id:{sample.get('_id')}")
                self.debug_count -= 1
            r_toks = r_toks[:max_toks_for_doc]
        r_ids = self.tokenizer.convert_tokens_to_ids(r_toks)

        input_ids = torch.tensor([q_ids + r_ids + [self.tokenizer.sep_token_id]], dtype=torch.int64)
        attention_mask = torch.tensor([[1] * input_ids.shape[1]], dtype=torch.int64)
        token_type_ids = torch.tensor([[0] * para_offset + [1] * (input_ids.shape[1]-para_offset)], dtype=torch.int64)
        item = {}
        item["encodings"] = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}
        item["label"] = torch.LongTensor([-1])
        item["question"] = q
        item["context"] = c
        item["answers"] = sample["answer"]
        item["index"] = index
        item["_id"] = str(index)
        item["para_offset"] = para_offset  # 1st tok after query ie [SEP]

        return item


class S1S2EvalDataset(Dataset):
    """ Stage 1 Para Reranker Model simplified Eval dataset
    Here instead of reader.reader_dataset.py since only used from rr_eval_truthfulqa.py so simpler..
    input samples: list of dict_keys(['question', 'answer', 'q_only', 'mc_options', 'context'])
    q_only key contains the base question minus mc options or other context
    """
    def __init__(self, args, tokenizer, samples, score_llm=True, model_type='s1'):
        self.score_llm = score_llm
        if score_llm:
            self.expl_key = 'context'       # llm-generated explanation/rationale/context including initial para for iirc
        else:
            self.expl_key = 'iter_context'  # iterator-generated context including init para for iirc            
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_c_len # overall max seq len not max len of context portion..
        self.max_q_len = args.max_q_len
        self.debug = args.debug
        self.debug_count = 3
        self.data = samples
        self.model_type = model_type
        print(f"Data size: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        # encode query [c+]q[+mc]
        q = sample['q_only'].strip()
        m = sample['mc_options'].strip()
        if m != '':
            m = ' ' + m
        q = q + m
        q_toks = self.tokenizer.tokenize(q)[:self.max_q_len]
        c = sample[self.expl_key].strip()  # context to score
        if c != '':
            if c[-1] != '.':
                c += '.'

        sents = split_into_sentences(c)
        pre_sents = []
        if self.model_type == 's1':
            for idx, sent in enumerate(sents):
                pre_sents.append("[unused1] " + sent.strip())
            c = GENERIC_TITLE + " " + " ".join(pre_sents)
        else:  #s2
            for idx, sent in enumerate(sents):
                pre_sents.append("[unused1] " + GENERIC_TITLE + ' | ' + sent.strip())
            c = " ".join(pre_sents)

        para_offset = len(q_toks) + 1 #  cls
        q_ids = [self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(q_toks)
        max_toks_for_doc = self.max_seq_len - para_offset - 1
        if max_toks_for_doc <= 2 and self.debug and self.debug_count > 0:
            print(f"Query too long: _id:{sample.get('_id')}")
            self.debug_count -= 1
        
        rat = NON_EXTRACTIVE_OPTIONS + c
        r_toks = self.tokenizer.tokenize(rat)
        if len(r_toks) > max_toks_for_doc:
            if self.debug and self.debug_count > 0:
                if len(r_toks) > 511:
                    print(f"RAT > 511 toks: index:{index}")
                print(f"Rat truncated. index:{index} _id:{sample.get('_id')}")
                self.debug_count -= 1
            r_toks = r_toks[:max_toks_for_doc]
        r_ids = self.tokenizer.convert_tokens_to_ids(r_toks)

        input_ids = torch.tensor([q_ids + r_ids + [self.tokenizer.sep_token_id]], dtype=torch.int64)
        attention_mask = torch.tensor([[1] * input_ids.shape[1]], dtype=torch.int64)
        token_type_ids = torch.tensor([[0] * para_offset + [1] * (input_ids.shape[1]-para_offset)], dtype=torch.int64)
        item = {}
        item["encodings"] = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}
        item["label"] = torch.LongTensor([-1])
        item["question"] = q
        item["context"] = c
        item["answers"] = sample["answer"]
        item["index"] = index
        item["_id"] = str(index)
        item["para_offset"] = para_offset  # 1st tok after query ie [SEP]

        return item



def predict_simple(eval_dataloader, model):
    """ Simple predict routine for scoring rationales
    """
    out_qids = []
    out_scores = []
    for batch in tqdm(eval_dataloader):
        #TJH batch = next(iter(eval_dataloader))
        # batch_to_feed = batch["net_inputs"]
        #batch = copy.deepcopy(batch_orig)
        batch_to_feed = move_to_cuda(batch["net_inputs"])
        batch_qids = batch["qids"]
        with torch.inference_mode():
            outputs = model(batch_to_feed)  # dict_keys(['rank_score'])
            scores = outputs["rank_score"]
            scores = scores.sigmoid().view(-1).tolist()  
        out_qids.extend(batch_qids)
        out_scores.extend(scores)
    return out_scores, out_qids



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


def batch_collate(samples, pad_id=0):
    if len(samples) == 0:
        return {}
    
    batch = {
        'input_ids': collate_tokens([s["encodings"]['input_ids'] for s in samples], pad_id),
        'attention_mask': collate_tokens([s["encodings"]['attention_mask'] for s in samples], 0),
        'label': collate_tokens([s["label"] for s in samples], -1),
        }
        
    # roberta does not use token_type_ids but electra does
    if "token_type_ids" in samples[0]["encodings"]:
        batch["token_type_ids"] = collate_tokens([s["encodings"]['token_type_ids']for s in samples], 0)

    batched = {"net_inputs": batch }
    
    # for eval only:
    if "question" in samples[0]:  
        batched["question"] = [s["question"] for s in samples]
        batched["context"] = [s["context"] for s in samples]
        batched["index"] = [s["index"] for s in samples]
        batched["qids"] = [s["_id"] for s in samples]
        batched["para_offsets"] = [s["para_offset"] for s in samples]
        batched["gold_answer"] = [s["answers"] for s in samples]

    return batched
