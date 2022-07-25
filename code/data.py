""" Single-dataset Loader

Author: Tim Hartill

Portions adapted from https://github.com/allenai/unifiedqa 
With other portions adapted from elsewhere as noted in comments
"""

import os
import json
import re
import string
import numpy as np

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

import utils
from eval_metrics import get_exact_match, selfsupervisedkey
from text_processing import ner, normalize_num, split_digits_special


#selfsupervisedkey = "_selfsvised"   # dataset names ending in this will be processed as self supervised
force_ans_start = '[#'              # if self supervised, can force a specific mask by using eg 'The rain in [#Spain#] lies mainly on the plain.'
force_ans_end = '#]'


def normalize_num_batch(instrlist, norm=''):
    """ Normalise numbers in a list of strings
    Set norm to '10e' to convert -123.45 to form '- 1 10e2 2 10e1 3 10e0 4 10e-1 5 10e-2'
    """
    outstrlist = []
    for instr in instrlist:
        outstr = normalize_num(instr, norm=norm)
        outstrlist.append(outstr)
    return outstrlist    


def pad_list(ids, attention_mask, max_length, pad_token_id):
    """ Pad list of token ids to max_length and adjust attention mask
    """
    numtoks = len(ids)
    if numtoks < max_length and pad_token_id is not None:
        padlen = max_length - numtoks
        padlist = [pad_token_id] * padlen
        ids = ids + padlist
        no_attention = [0] * padlen
        attention_mask = attention_mask + no_attention
    return ids, attention_mask


def get_word_starts(toks, specialchar = 'Ġ', bos_token='<s>'):
    """ Get the beginning of each word in a list of tokenised text
        Return list of word beginning indices into toks
    """
    word_starts = [i for (i,t) in enumerate(toks) if t[0]==specialchar or t[0] in string.punctuation]
    if toks[0] == bos_token: # don't want to mask a bos token 
        word_starts.pop(0)   
    if toks[word_starts[-1]] == specialchar: #space at end before newline
        word_starts.pop(-1)
    if word_starts[0] != 1:  # first non bos token is always a word start
        word_starts = [1] + word_starts
    return word_starts


def find_sub_list(sublst1, sublst2, lst):
    """ Return start/end indices of all occurences of sublist in list
        Note: Can't tell whether the tokens match with/out a preceding space so must try both ways ie sublst1 & sublst2
    """
    results=[]
    sll=len(sublst1)
    for ind in (i for i,e in enumerate(lst) if e==sublst1[0]):
        if lst[ind:ind+sll]==sublst1:
            results.append((ind,ind+sll))
    sll=len(sublst2)
    for ind in (i for i,e in enumerate(lst) if e==sublst2[0]):
        if lst[ind:ind+sll]==sublst2:
            results.append((ind,ind+sll))  
    results = list(set(results))
    new_results = []
    for l in results:
        new_results.append( list(l) )
    return new_results 


def map_ners(toks, ners, tokenizer, verbose = False):
    """ Map list of NERs previously identified on raw text to token ids
    """
    unique_ner = list(set([ w.strip(string.punctuation+' ') for w in ners ]))
    unique_ner = [w.strip() for w in unique_ner if w.strip() != '']
    tok_map = []
    final_ner = []
    for n in unique_ner:
        ner_txt_tok = tokenizer.tokenize(n)
        ner_txt_tok2 = tokenizer.tokenize(' ' + n)
        found_list = find_sub_list(ner_txt_tok, ner_txt_tok2, toks)
        if verbose: 
            print(f"Orig: {n}") 
            for tok_start, tok_end in found_list:
                print(f"tokens: {toks[tok_start:tok_end]}")
            if len(found_list) == 0:
                print(f"NOT FOUND: {n} 1:{ner_txt_tok} 2:{ner_txt_tok2}")
        if len(found_list) > 0:
            tok_map.append(found_list)
            final_ner.append(n)
    return final_ner, tok_map


def manual_encode(instr, tokenizer, args, truncation=True, max_length=512, 
                  pad=True, specialchar='Ġ', bos_token='', selfsupervised=False):
    """ Manually encode a string for Bart, Roberta, GPT2 or T5
    Note: If you call .tokenize() with a string that generates more tokens than the 
          max seq len of the model you get a warning which you can safely ignore..
    """
    if selfsupervised:
        ans_start_idx = instr.find(force_ans_start)
        if ans_start_idx == -1:
            ners = ner(instr) 
        else: # force a particular span to be masked
            ans_start_idx += len(force_ans_start)
            ans_end_idx = instr.find(force_ans_end)
            ners = [instr[ans_start_idx:ans_end_idx]]
            instr = instr.replace(force_ans_start, '').replace(force_ans_end, '')
    else:
        ners = []
        ners_ids = []
    if args.do_lowercase:
        instr = instr.lower()
        if selfsupervised:
            ners = [n.lower() for n in ners]
    toks = tokenizer.tokenize(instr)
#    if args.indiv_digits:  # superceded in favor of simply adding digits as special tokens to the tokenizer
#        toks = split_digits_special(toks, special=specialchar)
    if selfsupervised:
        wstarts = get_word_starts(toks, specialchar, bos_token)
        ners, ners_ids = map_ners(toks, ners, tokenizer)
    else:
        wstarts = []
    ids = tokenizer.convert_tokens_to_ids(toks)
    if tokenizer.bos_token_id is not None:
        ids = [tokenizer.bos_token_id] + ids
        if selfsupervised:
            for i in range(len(wstarts)): 
                wstarts[i] += 1
            for i in range(len(ners_ids)):
                for j in range(len(ners_ids[i])):
                    ners_ids[i][j][0] += 1
                    ners_ids[i][j][1] += 1
    numtoks = len(ids)
    if truncation and numtoks > max_length-1:
        ids = ids[:max_length-1]
        if selfsupervised:
            wstarts = [w for w in wstarts if w < max_length]
            new_ners_ids = []
            for i, item_ids in enumerate(ners_ids):
                new_item_id_list = []
                for j, nid in enumerate(item_ids):
                    if nid[1] < max_length: 
                        new_item_id_list.append(nid)
                if new_item_id_list:
                    new_ners_ids.append(new_item_id_list)
            ners_ids = new_ners_ids
        
    ids = ids + [tokenizer.eos_token_id]
    numtoks = len(ids)
    if not selfsupervised:
        attention_mask = [1] * numtoks
    else:
        attention_mask = []
    if pad and numtoks < max_length and tokenizer.pad_token_id is not None and not selfsupervised:
        ids, attention_mask = pad_list(ids, attention_mask, max_length, tokenizer.pad_token_id)
    return ids, attention_mask, wstarts, ners_ids


def manual_batch_encode(instrlist, tokenizer, logger, args, selfsupervised, metadata,
                        truncation=True, max_length=512, pad=False):
    """ Manually encode a list of strings for T5, BART, Roberta or GPT2
    Returns dict {'input_ids': [ [], [] ],
                  'attention_mask': [ [], [] ],
                  'word_starts': []}
    """
    if args.append_another_bos and tokenizer.bos_token_id is None:
        logger.info("Tokenizer has no bos token so ignoring --append_another_bos flag.")
        bos_token = ''  # T5 doesnt have BOS token
    elif args.append_another_bos:
        bos_token = tokenizer.bos_token + ' '  # gpt2 bos token = eos token...
    else:
        bos_token = ''

    specialchar = 'Ġ'
    if 't5' in str(tokenizer.__class__):
        specialchar = '▁'

    if args.norm_numbers:
        norm = ''
        if args.norm_10e:
            norm = '10e'
        instrlist = normalize_num_batch(instrlist, norm=norm)
        
    if args.strip_single_quotes:   # Added for T5
        instrlist = [re.sub("'(.*)'", r"\1", s) for s in instrlist]
    
    if args.append_another_bos:
        instrlist = [bos_token + s if s != '' else s for s in instrlist]  # Don't add bos to nonexistent answer if self supervised
    bos_token = bos_token.strip()
    
    outdict = {}
    input_ids_list = []
    attention_mask_list = []
    word_starts_list = []
    ners_ids_list = []
    meta_index = 0
    for i, instr in enumerate(instrlist):
        if i >= metadata[meta_index][1]:
            meta_index += 1    
        if instr != '':
            input_ids, attention_mask, wstarts, ners_ids = manual_encode(instr, tokenizer, args,
                                                      truncation=truncation,
                                                      max_length=max_length,
                                                      pad=pad, specialchar=specialchar,
                                                      bos_token=bos_token,
                                                      selfsupervised=selfsupervised[meta_index])
        else:
            input_ids = []
            attention_mask = []
            wstarts = []
            ners_ids = []
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        word_starts_list.append(wstarts)
        ners_ids_list.append(ners_ids)
    outdict['input_ids'] = input_ids_list
    outdict['attention_mask'] = attention_mask_list
    outdict['word_starts'] = word_starts_list
    outdict['ners_ids'] = ners_ids_list
    return outdict

      

class QAData(object):

    def __init__(self, logger, args, data_path, is_training):
        self.data_path = data_path
        if args.debug:
            self.data_path = data_path.replace("train", "dev")
        if "/test" in self.data_path:
            self.data_type = "test"
        elif "/dev" in self.data_path:
            self.data_type = "dev"
        elif "/train" in self.data_path:
            self.data_type = "train"
        else:
            raise NotImplementedError()
        assert self.data_path.endswith(".tsv"), "data file has to be in tsv format."
        if selfsupervisedkey in data_path:
            self.selfsupervised = [True]
        else:
            self.selfsupervised = [False]
        self.data = []
        with open(self.data_path, "r") as f:
            cnt = 0
            invalid_lines = 0
            for line in f:
                try:
                    if self.selfsupervised[-1]:
                        answer = ""
                        question = line.strip()
                    else:
                        question, answer = line.split("\t")
                        if answer.lstrip().startswith(utils.MULTI_ANS_SEP): # #!# answer 1#!# answer 2 #!# -> ['answer 1','answer 2']
                            answer = answer.strip().strip(utils.MULTI_ANS_SEP).split(utils.MULTI_ANS_SEP)
                            answer = [a + '\n' for a in answer]
                        else:
                            answer = [answer] # note always ends in \n                     
                except Exception:
                    invalid_lines += 1
                    continue
                self.data.append({
                    "id": "{}-{}".format(self.data_type, cnt),
                    "question": question,
                    "answer": answer
                })
                cnt += 1
            if invalid_lines > 0:
                print ("# invalid lines: {}".format(invalid_lines))

        if args.debug:
            self.data = self.data[:40]
        assert type(self.data)==list
        assert all(["id" in d for d in self.data]), self.data[0].keys()
        if type(self.data[0]["id"])==int:
            for i in range(len(self.data)):
                self.data[i]["id"] = str(self.data[i]["id"])

        self.index2id = {i:d["id"] for i, d in enumerate(self.data)}
        self.id2index = {d["id"]:i for i, d in enumerate(self.data)}
        self.is_training = is_training
        self.load = not args.debug
        self.logger = logger
        self.args = args
        self.metric = "EM"
        self.max_input_length = self.args.max_input_length
        self.tokenizer = None
        self.dataset = None
        self.dataloader = None
        self.cache = None
        self.preprocessed_path = None  # set in load_dataset

    def __len__(self):
        return len(self.data)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens,
                                     skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True).strip()

    def decode_batch(self, tokens):
        return [self.decode(_tokens) for _tokens in tokens]

    def flatten(self, answers):  # Note: With input [['abc\n', 'sample1 ans2\n'],[...]] returns ['abc\n', '...']
                                # and originally metadata [(0, 1), (1, 2), (2, 3), ..]
                                # changed metadata to [(0, len(answers))] to match uqa format
        """ For training each sample must only have 1 answer - others will be ignored
            For dev/eval - can have multiple answers for EM and F1 calc which come from self.data.answer but only the 1st answer is tokenised (tokenised dev answer not actually used for anything)
        """
        new_answers, metadata = [], []
        for answer in answers:
            assert type(answer)==list
            new_answers.append(answer[0])  # Note: can still have multiple answers for eval but only 1st one will be tokenised. 
        metadata = [(0, len(new_answers))]
        return new_answers, metadata

    def load_dataset(self, tokenizer, do_return=False, load_preprocessed=True):
        if not load_preprocessed: 
            self.load = False  # don't load or save tokenised data to file
        self.tokenizer = tokenizer
        postfix = tokenizer.__class__.__name__.replace("zer", "zed")
        preprocessed_path = os.path.join(
            "/".join(self.data_path.split("/")[:-1]),
            self.data_path.split("/")[-1].replace(
                ".tsv" if self.data_path.endswith(".tsv") else ".json",
                "-v3-{}{}{}{}{}{}-{}.json".format(
                    "-uncased" if self.args.do_lowercase else "",
                    "-xbos" if (self.args.append_another_bos and self.tokenizer.bos_token_id is not None) else "",
                    "-squote" if (self.args.strip_single_quotes) else "",
                    "-idigits" if (self.args.indiv_digits) else "",
                    "-nnorm" if (self.args.norm_numbers) else "",
                    "-10e" if (self.args.norm_10e) else "",
                    postfix)))
        
        self.preprocessed_path = preprocessed_path

        if self.load and os.path.exists(preprocessed_path):
            self.logger.info("Loading pre-tokenized data from {}".format(preprocessed_path))
            with open(preprocessed_path, "r") as f:
                input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, \
                    metadata, word_starts, ners_ids = json.load(f)
        else:
            print ("Start tokenizing...")
            
            questions = [d["question"] if d["question"].endswith("?") else d["question"]+"?"
                        for d in self.data]
            answers = [d["answer"] for d in self.data]
            answers, metadata = self.flatten(answers)  # "flatten" means "take 1st answer only". For training, dev only tokenise 1st answer. For dev tokenised answer not actually used for anything..

            question_input = manual_batch_encode(questions, 
                                                 self.tokenizer,
                                                 self.logger,
                                                 self.args,
                                                 self.selfsupervised,
                                                 metadata,
                                                 truncation=True,
                                                 pad=False,
                                                 max_length=self.args.max_input_length)
            answer_input = manual_batch_encode(answers, 
                                                 self.tokenizer,
                                                 self.logger,
                                                 self.args,
                                                 self.selfsupervised,
                                                 metadata,
                                                 truncation=True,
                                                 pad=False,
                                                 max_length=self.args.max_output_length)

            word_starts = question_input["word_starts"] 
            ners_ids = question_input["ners_ids"]
            input_ids, attention_mask = question_input["input_ids"], question_input["attention_mask"]
            decoder_input_ids, decoder_attention_mask = answer_input["input_ids"], answer_input["attention_mask"]
            
            if self.load:
                with open(preprocessed_path, "w") as f:
                    json.dump([input_ids, attention_mask,
                               decoder_input_ids, decoder_attention_mask,
                               metadata, word_starts, ners_ids], f)
                self.logger.info("Saved tokenised data to {}".format(preprocessed_path))

        self.dataset = MyQADataset(input_ids, attention_mask,
                                         decoder_input_ids, decoder_attention_mask, self.args, self.data,
                                         metadata=metadata,
                                         is_training=self.is_training,
                                         tokenizer=self.tokenizer,
                                         selfsupervised = self.selfsupervised,
                                         word_starts = word_starts,
                                         ners_ids=ners_ids)
        self.logger.info("Loaded {} examples from {} data".format(len(self.dataset), self.data_type))

        if do_return:
            return self.dataset

    def load_dataloader(self, do_return=False):
        self.dataloader = MyDataLoader(self.args, self.dataset, self.is_training)
        if do_return:
            return self.dataloader

    def evaluate(self, predictions):
        assert len(predictions)==len(self), (len(predictions), len(self))
        ems = []
        for (prediction, dp) in zip(predictions, self.data):
            ems.append(get_exact_match(prediction, dp["answer"]))
        return ems

    def save_predictions(self, predictions):
        assert len(predictions)==len(self), (len(predictions), len(self))
        #prediction_dict = {dp["id"]:prediction for dp, prediction in zip(self.data, predictions)}
        save_path = os.path.join(self.args.output_dir, "{}predictions.json".format(self.args.prefix))
        with open(save_path, "w") as f:
            json.dump(predictions, f)
        self.logger.info("Saved prediction in {}".format(save_path))
        
    def load_predictions(self):
        """ Load predictions from file of form ['pred1 test', 'pred 2 text' ...]
        """
        load_path = os.path.join(self.args.output_dir, "{}predictions.json".format(self.args.prefix))
        if os.path.exists(load_path):
            predictions = json.load(open(load_path))
            assert len(predictions)==len(self), f"Invalid prediction file length. Expected {len(self)} but got {len(predictions)}"
            self.logger.info("Successfully loaded predictions from {}".format(load_path))
            
        else:
            self.logger.info("Error: Predictions file doesnt exist. Run with --do_predict set to generate the file {}".format(load_path))
            assert os.path.exists(load_path), "Exiting since predictions file doesnt exist."
        return predictions    
            

def get_spans(tok_idxs, toks_to_mask=0.11, avg_span_len=2, sd=0.75):
    """ Calculate number and length of spans for given input seq length
    """
    num_toks = len(tok_idxs)
    num_spans = int( (num_toks * toks_to_mask) / avg_span_len) + 1
    span_lengths = np.random.normal(avg_span_len, scale=sd, size=num_spans).round().astype('int')
    span_lengths = np.clip(span_lengths, 1, avg_span_len+4)    
    return span_lengths


def merge_intervals(in_list):
    """ Merge overlapping intervals in a list
    """
    in_list.sort(key=lambda interval: interval[0])
    merged = [in_list[0]]
    for current in in_list:
        previous = merged[-1]
        if current[0] <= previous[1]:
            previous[1] = max(previous[1], current[1])
        else:
            merged.append(current)
    return merged


def wwsc_select_spans(tok_idxs, span_lengths, word_starts):
    """ Convert a set of span lengths into actual start/end token positions
    """
    num_toks = len(tok_idxs)
    num_words = len(word_starts)
    if num_words == 0:
        return []
    replace_spans = []
    for length in span_lengths:
        span_start_idx = np.random.choice(num_words)
        span_start = word_starts[span_start_idx]
        if span_start + length > num_toks:
            length = num_toks - span_start
        else:
            for next_wordstart in word_starts[span_start_idx+1:]:
                if next_wordstart >= span_start+length:
                    length = next_wordstart - span_start
                    break
        span_end = span_start + length
        replace_spans.append( [span_start, span_end]  )
    replace_spans = merge_intervals(replace_spans)  # aggregate overlaps
    return replace_spans


def mask_words(tok_idxs, replace_spans, mask_seq):
    """ Given a list of token indices,  + an array of spans [[start1, end1], [start2, end2], ...]
        + a list of mask substitutions
        return a masked version of toks plus the list of masked spans 
    """
    replaced_toks = []
    tmp_tok_idxs = tok_idxs.copy()
    ctr = 0
    for replace_span in replace_spans:
        replaced_toks.append( mask_seq[ctr] + tok_idxs[replace_span[0]:replace_span[1]] )
        ctr += 1
        if ctr > 18:  # use mask_seq[19] to as answer end indicator
            ctr = 0
        first = True
        for i in range(replace_span[0], replace_span[1]):
            if first:
                tmp_tok_idxs[i] = -8888
                first = False
            else:
                tmp_tok_idxs[i] = -9999
    new_tok_idxs = []
    ctr = 0
    for tok in tmp_tok_idxs:
        if tok != -9999:
            if tok == -8888:
                new_tok_idxs.extend(mask_seq[ctr])
                ctr += 1
                if ctr > 18:  # use mask_seq[19] to as answer end indicator
                    ctr = 0
            else:    
                new_tok_idxs.append(tok)
    return new_tok_idxs, replaced_toks


def self_supervise(args, tok_idxs, word_starts, ners_ids, mask_seq, nq_token_ids, bos_token_id, eos_token_id):
    """ Mask text and return masked input and list of masked spans
    """
    if len(ners_ids) > 0 and np.random.rand() < args.ssm_prob:
        ner_idx = np.random.choice(len(ners_ids))
        ner_pos = np.random.choice(len(ners_ids[ner_idx]))
        replace_spans = [ ners_ids[ner_idx][ner_pos] ]
        new_tok_idxs, replaced_toks = mask_words(tok_idxs, replace_spans, mask_seq)
    else:  #Whole Word Span Corruption
        span_lengths = get_spans(tok_idxs, toks_to_mask=args.wwsc_toks_to_mask, 
                                 avg_span_len=args.wwsc_avg_span_len, sd=args.wwsc_span_len_sd)
        replace_spans = wwsc_select_spans(tok_idxs, span_lengths, word_starts)
        new_tok_idxs, replaced_toks = mask_words(tok_idxs, replace_spans, mask_seq)
    if replaced_toks == []:
        replaced_toks = [mask_seq[0] + nq_token_ids]
    replaced_toks.append(mask_seq[19])
    new_replaced_toks = []
    for r in replaced_toks:
        new_replaced_toks.extend(r)
    if bos_token_id is not None:
        if args.append_another_bos:
            new_replaced_toks = [bos_token_id] + new_replaced_toks
        new_replaced_toks = [bos_token_id] + new_replaced_toks        
    new_replaced_toks.append(eos_token_id)    
    if len(new_tok_idxs) > args.max_input_length:
        print(f"Warning: self_supervise: Input Truncated {len(new_tok_idxs)} to {args.max_input_length}")
        new_tok_idxs = new_tok_idxs[:args.max_input_length-1] + [eos_token_id]
    if len(new_replaced_toks) > args.max_output_length:
        print(f"Warning: self_supervise: Answer Truncated {len(new_replaced_toks)} to {args.max_output_length}")
        new_replaced_toks = new_replaced_toks[:args.max_output_length-1] + [eos_token_id]
    attention_mask = [1] * len(new_tok_idxs)
    decoder_attention_mask = [1] * len(new_replaced_toks)
    return new_tok_idxs, attention_mask, new_replaced_toks, decoder_attention_mask


class MyQADataset(Dataset):
    def __init__(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, args, data,
                 metadata=None, is_training=False, tokenizer=None, selfsupervised=None, 
                 word_starts=None, ners_ids=None):
        self.args = args
        self.parent_data = data
        self.tokenizer = tokenizer
        if tokenizer is not None and tokenizer.pad_token_id is not None:
            self.pad_token_id = tokenizer.pad_token_id
        else:
            self.pad_token_id = -100
        if 't5' in str(tokenizer.__class__):
            self.bos_token_id = None
            self.mask_token_id = 32099   # '<extra_id_0>
            self.mask_token = '<extra_id_0>'
            self.mask_seq = [ '<extra_id_0>', '<extra_id_1>', '<extra_id_2>', '<extra_id_3>', '<extra_id_4>', '<extra_id_5>',
                              '<extra_id_6>', '<extra_id_7>', '<extra_id_8>', '<extra_id_9>', '<extra_id_10>', '<extra_id_11>',
                              '<extra_id_12>', '<extra_id_13>', '<extra_id_14>', '<extra_id_15>', '<extra_id_16>', '<extra_id_17>',
                              '<extra_id_18>', '<extra_id_19>']
        else:
            self.bos_token_id = tokenizer.bos_token_id
            self.mask_token_id = tokenizer.mask_token_id
            self.mask_token = tokenizer.mask_token
            self.mask_seq = [tokenizer.mask_token] * 20
        if args.add_mask_char != 'NONE':
            self.mask_seq = [m+args.add_mask_char for m in self.mask_seq]
        if args.add_mask_ctr:
            self.mask_seq = [m+str(i) for i, m in enumerate(self.mask_seq)]
        self.mask_seq = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(m)) for m in self.mask_seq]
        self.eos_token_id = tokenizer.eos_token_id
        self.no_question_label = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("no mask"))        
        self.selfsupervised = selfsupervised
        self.metadata = [(0, len(input_ids))]  #override historical metadata setup
        self.input_ids = input_ids              # torch.LongTensor(input_ids)
        self.attention_mask = attention_mask    # torch.LongTensor(attention_mask)
        self.decoder_input_ids = decoder_input_ids              # torch.LongTensor(decoder_input_ids)
        self.decoder_attention_mask = decoder_attention_mask    # torch.LongTensor(decoder_attention_mask)
        self.word_starts = word_starts
        self.ners_ids = ners_ids
        self.is_training = is_training

        assert len(self.input_ids)==len(self.attention_mask)==len(self.decoder_input_ids)==len(self.decoder_attention_mask)==len(self.word_starts)==len(self.ners_ids)
        assert len(self.input_ids)==metadata[-1][-1]
        

    def __len__(self):
        return self.metadata[-1][-1]   # num questions

    def __getitem__(self, idx):
        ssvise = self.selfsupervised[0]
        if not self.is_training:
            if ssvise:
                input_ids, attention_mask, decoder_input_ids, decoder_attention_mask = self_supervise(self.args, 
                                                                                                      self.input_ids[idx], 
                                                                                                      self.word_starts[idx], 
                                                                                                      self.ners_ids[idx], 
                                                                                                      self.mask_seq, 
                                                                                                      self.no_question_label,
                                                                                                      self.bos_token_id,
                                                                                                      self.eos_token_id)
                self.parent_data[idx]['answer'] = self.tokenizer.decode(decoder_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                input_ids, attention_mask = pad_list(input_ids, attention_mask, self.args.max_input_length, self.pad_token_id)
            else:
                input_ids, attention_mask = pad_list(self.input_ids[idx], self.attention_mask[idx],
                                                     self.args.max_input_length, self.pad_token_id)
            input_ids = torch.LongTensor(input_ids)
            attention_mask = torch.LongTensor(attention_mask)
            return input_ids, attention_mask
        
        if ssvise:
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask = self_supervise(self.args, 
                                                                                                  self.input_ids[idx], 
                                                                                                  self.word_starts[idx], 
                                                                                                  self.ners_ids[idx], 
                                                                                                  self.mask_seq, 
                                                                                                  self.no_question_label,
                                                                                                  self.bos_token_id,
                                                                                                  self.eos_token_id)
            input_ids, attention_mask = pad_list(input_ids, attention_mask, self.args.max_input_length, self.pad_token_id)
            decoder_input_ids, decoder_attention_mask = pad_list(decoder_input_ids, decoder_attention_mask, self.args.max_output_length, self.pad_token_id)

        else:
            input_ids, attention_mask = pad_list(self.input_ids[idx], self.attention_mask[idx],
                                                 self.args.max_input_length, self.pad_token_id)
            decoder_input_ids, decoder_attention_mask = pad_list(self.decoder_input_ids[idx], self.decoder_attention_mask[idx],
                                                                 self.args.max_output_length, self.pad_token_id)
        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor(attention_mask)
        decoder_input_ids = torch.LongTensor(decoder_input_ids)
        decoder_attention_mask = torch.LongTensor(decoder_attention_mask)
        return input_ids, attention_mask, decoder_input_ids, decoder_attention_mask


class MyDataLoader(DataLoader):

    def __init__(self, args, dataset, is_training):
        if is_training:
            sampler=RandomSampler(dataset)
            batch_size = args.train_batch_size
        else:
            sampler=SequentialSampler(dataset)
            batch_size = args.predict_batch_size
        super(MyDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size)


