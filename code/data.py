""" Single-dataset Loader

Author: Tim Hartill

Adapted from https://github.com/allenai/unifiedqa 
With other portions adapted from elsewhere as noted in comments
"""

import os
import json
import re
import string
import numpy as np

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

from eval_metrics import get_exact_match

from w2n import word_to_num   # from https://github.com/ag1988/injecting_numeracy/blob/master/pre_training/gen_bert/create_examples_n_features.py
import spacy
nlp = spacy.load("en_core_web_sm")
selfsupervisedkey = "_selfsvised"  # dataset names ending in this will be processed as self supervised


# from https://github.com/castorini/transformers-arithmetic/blob/main/main.py
def convert_to_base(num: int, base: int, numerals="0123456789abcdefghijklmnopqrstuvwxyz") -> str:
    return ((num == 0) and numerals[0]) or (
        convert_to_base(num // base, base, numerals).lstrip(numerals[0]) + numerals[num % base])


# adapted from https://github.com/castorini/transformers-arithmetic/blob/main/main.py
def convert_to_10ebased(number: str, split_type: str=None, invert_number: bool=False) -> str:
    signal = None
    if number[0] == '-':
        signal = '-'
        number = number[1:]

    digitpos = number.find('.')
    if digitpos != -1:
        number = number.replace('.', '')
        i = (len(number) - digitpos) * -1
    else:
        i = 0
        
    output = []
    for digit in number[::-1]:
        if split_type is None:
            output.append('10e' + str(i))
        elif split_type == 'underscore':
            output.append('10e_' + str(i))
        elif split_type == 'character':
            output.append(' '.join('D' + str(i) + 'E'))
        else:
            raise Exception(f'Wrong split_type: {split_type}')
        output.append(digit)
        i += 1

    if signal:
        output.append(signal)

    # The output is already inverted. If we want it to _not_ be inverted, then we invert it.
    if not invert_number:
        output = output[::-1]
    return ' '.join(output)


#Adapted from https://github.com/ag1988/injecting_numeracy/blob/master/pre_training/gen_bert/create_examples_n_features.py    
def convert_word_to_number(word: str):
    """
    Returns number if convertable from word string otherwise None
    """
    # strip all punctuations from the sides of the word, except for the negative sign
    punctuations = string.punctuation.replace('-', '')
    if word[0] == '.':
        punctuations = punctuations.replace('.', '')
    word = word.strip(punctuations)
    # some words may contain the comma as deliminator
    word = word.replace(",", "")
    # word2num will convert hundred, thousand ... to number, but we skip it.
    if word in ["hundred", "thousand", "million", "billion", "trillion"]:
        return None
    try:
        number = word_to_num(word)
    except ValueError:
        try:
            number = int(word)
        except ValueError:
            try:
                number = float(word)
            except ValueError:
                number = None
    return number


def normalize_num(instr, norm=''):
    """ Normalise numbers found in input string and return normalised string
    """
    doc = nlp(instr)  #spacy tokenization
    newtext = []
    for token in doc:
        if token.pos_ == 'NUM' or (len(token.text)>1 and set(token.text).issubset(set('0123456789,-.'))):
            norm_word = convert_word_to_number(token.text)
            if norm_word is not None:
                norm_num = str(norm_word)
                if norm == '10e':
                    norm_num = convert_to_10ebased(norm_num)
                if token.text_with_ws[-1] == ' ':
                    norm_num += ' '
                newtext.append(norm_num)
            else:
                newtext.append(token.text_with_ws)        
        else:
            newtext.append(token.text_with_ws)
    outstr = ''.join(newtext)   
    return outstr    


def normalize_num_batch(instrlist, norm=''):
    """ Normalise numbers in a list of strings
    Set norm to '10e' to convert -123.45 to form '- 1 10e2 2 10e1 3 10e0 4 10e-1 5 10e-2'
    """
    outstrlist = []
    for instr in instrlist:
        outstr = normalize_num(instr, norm=norm)
        outstrlist.append(outstr)
    return outstrlist    


# Adapted from BERT/wordpiece version split_digits from https://github.com/ag1988/injecting_numeracy/blob/master/pre_training/gen_bert/create_examples_n_features.py:
def split_digits_special(wps, special='Ġ'): #-> List[str]:
    """
    Further split numeric tokens accommodating arbitrary special char 
    For t5 special='▁'  (not underscore)
    For bart/roberta special='Ġ'
    eg tokenizer_bart.tokenize('124567890')-> ['12', '45', '678', '90']
      split_digits_special(tokenizer_bart.tokenize('124567890'), special='Ġ') -> ['1', '2', '4', '5', '6', '7', '8', '9', '0']
    eg2 split_digits_special(tokenizer_t5.tokenize('the rain in 124567890 99 999 is similar to a55.'), special='▁')
    """
    toks = []
    for wp in wps:
        if set(wp).issubset(set(special+'0123456789.-$,^')) and set(wp) != {special}: # numeric wp - split digits
            for i, dgt in enumerate(list(wp.replace(special, ''))):
                prefix = special if (wp.startswith(special) and i == 0) else ''
                toks.append(prefix + dgt)
        else:
            toks.append(wp)
    return toks


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


def ner(instr, verbose=False):
    """ Perform named entity recognition on text and return a list of named entities, numbers, dates etc
    """
    ner_list = []
    doc = nlp(instr.replace('\\n \n', ''))    
    for ent in doc.ents:
        if verbose: print(ent.text, '"' + ent.text_with_ws + '"', ent.start_char, ent.end_char, ent.label_)
        ner_list.append(ent.text_with_ws)
        #ner_list.append( {'txt_with_ws': ent.text_with_ws, 'start':ent.start_char, 'end': ent.end_char, 'type': ent.label_} )
    return ner_list


def find_sub_list(sublst1, sublst2, lst):
    """ Return start/end indices of all occurences of sublist in list
        Note: Can't tell whether the tokens match with/out a preceding space so must try both ways
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
    unique_ner = list(set([ w.strip(string.punctuation+' ').strip() for w in ners ]))
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
        ners = ner(instr) 
    else:
        ners = []
        ners_ids = []
    if args.do_lowercase:
        instr = instr.lower()
        if selfsupervised:
            ners = [n.lower() for n in ners]
    toks = tokenizer.tokenize(instr)
    if args.indiv_digits:
        toks = split_digits_special(toks, special=specialchar)
    if selfsupervised:
        wstarts = get_word_starts(toks, specialchar, bos_token)
        ners, ners_ids = map_ners(toks, ners, tokenizer)
    else:
        wstarts = []
    ids = tokenizer.convert_tokens_to_ids(toks)
    if tokenizer.bos_token_id is not None:
        ids = [tokenizer.bos_token_id] + ids
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
    attention_mask = [1] * numtoks
    if numtoks < max_length and pad and tokenizer.pad_token_id is not None:
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
                except Exception:
                    invalid_lines += 1
                    continue
                self.data.append({
                    "id": "{}-{}".format(self.data_type, cnt),
                    "question": question,
                    "answer": [answer]
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

    def flatten(self, answers):  # Note: With input [['abc\n'],[...]] returns ['abc\n', '...']
                                # and originally metadata [(0, 1), (1, 2), (2, 3), ..]
                                # changed metadata to [(0, len(answers))] to match uqa format
        new_answers, metadata = [], []
        for answer in answers:
            assert type(answer)==list
            #metadata.append((len(new_answers), len(new_answers)+len(answer)))
            new_answers += answer  # Note: can only have 1 answer now
        metadata = [(0, len(new_answers))]
        return new_answers, metadata

    def load_dataset(self, tokenizer, do_return=False):
        self.tokenizer = tokenizer
        postfix = tokenizer.__class__.__name__.replace("zer", "zed")
        preprocessed_path = os.path.join(
            "/".join(self.data_path.split("/")[:-1]),
            self.data_path.split("/")[-1].replace(
                ".tsv" if self.data_path.endswith(".tsv") else ".json",
                "-v2-{}{}{}{}{}{}-{}.json".format(
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
            answers, metadata = self.flatten(answers)

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
                                         decoder_input_ids, decoder_attention_mask, self.args,
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


def mask_words(tok_idxs, replace_spans, mask_token):
    """ Given a list of token indices,  + an array of spans [[start1, end1], [start2, end2], ...]
        return a masked version of toks plus the list of masked spans 
    """
    replaced_toks = []
    tmp_tok_idxs = tok_idxs.copy()
    for replace_span in replace_spans:
        replaced_toks.append( tok_idxs[replace_span[0]:replace_span[1]] )
        first = True
        for i in range(replace_span[0], replace_span[1]):
            if first:
                tmp_tok_idxs[i] = mask_token
                first = False
            else:
                tmp_tok_idxs[i] = -999999
    new_tok_idxs = []
    for tok in tmp_tok_idxs:
        if tok != -999999:
            new_tok_idxs.append(tok)
    return new_tok_idxs, replaced_toks


def self_supervise(args, tok_idxs, word_starts, ners_ids, mask_token_id, unk_token_id):
    """ Mask text and return masked input and list of masked spans
    """
    if len(ners_ids) > 0 and np.random.rand() > args.ssm_prob:
        ner_idx = np.random.choice(len(ners_ids))
        ner_pos = np.random.choice(len(ners_ids[ner_idx]))
        replace_spans = [ ners_ids[ner_idx][ner_pos] ]
        new_tok_idxs, replaced_toks = mask_words(tok_idxs, replace_spans, mask_token=mask_token_id)
    else:  #Whole Word Span Corruption
        span_lengths = get_spans(tok_idxs, toks_to_mask=args.wwsc_toks_to_mask, 
                                 avg_span_len=args.wwsc_avg_span_len, sd=args.wwsc_span_len_sd)
        replace_spans = wwsc_select_spans(tok_idxs, span_lengths, word_starts)
        new_tok_idxs, replaced_toks = mask_words(tok_idxs, replace_spans, mask_token=mask_token_id)
    if replaced_toks == []:
        replaced_toks = [[unk_token_id]]
    #TODO add answer formatting + padding + make attn masks
    return new_tok_idxs, replaced_toks


class MyQADataset(Dataset):
    def __init__(self,
                 input_ids, attention_mask,
                 decoder_input_ids, decoder_attention_mask, args,
                 metadata=None,
                 is_training=False, tokenizer=None, selfsupervised=None, 
                 word_starts=None, ners_ids=None):
        self.args = args
        self.tokenizer = tokenizer
        if tokenizer is not None and tokenizer.pad_token_id is not None:
            self.pad_token_id = tokenizer.pad_token_id
        else:
            self.pad_token_id = -100
        if 't5' in str(tokenizer.__class__):
            self.bos_token_id = None
            self.mask_token_id = 32099   # '<extra_id_0>
        else:
            self.bos_token_id = tokenizer.bos_token_id
            self.mask_token_id = tokenizer.mask_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.unk_token_id =tokenizer.unk_token_id
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
        return self.metadata[-1][-1]  #len(self.in_metadata)  # num questions

    def __getitem__(self, idx):
        objective = self.selfsupervised[0]
        if not self.is_training:
            if objective: #TODO write generated txt labels into data + calc masked input
                input_ids, replaced_toks =  self_supervise(self.args, self.input_ids[idx], 
                                                           self.word_starts[idx], self.ners_ids[idx], self.mask_token_id, self.unk_token_id)
                
            else:
                input_ids, attention_mask = pad_list(self.input_ids[idx], self.attention_mask[idx],
                                                     self.args.max_input_length, self.pad_token_id)
            input_ids = torch.LongTensor(input_ids)
            attention_mask = torch.LongTensor(attention_mask)
            return input_ids, attention_mask
        
        if objective:  #TODO calc both masked input + generated labels as toks
            pass
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


