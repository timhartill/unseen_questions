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
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

from w2n import word_to_num   # from https://github.com/ag1988/injecting_numeracy/blob/master/pre_training/gen_bert/create_examples_n_features.py
import spacy
nlp = spacy.load("en_core_web_sm")


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


def manual_encode(instr, tokenizer, truncation=True, max_length=512, 
                  pad=True, indiv_digits=True):
    """ Manually encode a string for Bart, Roberta, GPT2 or T5
    Note: If you call .tokenize() with a string that generates more tokens than the 
          max seq len of the model you get a warning which you can safely ignore..
    """
    toks = tokenizer.tokenize(instr)
    if indiv_digits:
        specialchar = 'Ġ'
        if 't5' in str(tokenizer.__class__):
            specialchar = '▁'
        toks = split_digits_special(toks, special=specialchar)
    ids = tokenizer.convert_tokens_to_ids(toks)
    if tokenizer.bos_token_id is not None:
        ids = [tokenizer.bos_token_id] + ids
    numtoks = len(ids)
    if truncation and numtoks > max_length-1:
        ids = ids[:max_length-1]
    ids = ids + [tokenizer.eos_token_id]
    numtoks = len(ids)
    attention_mask = [1] * numtoks
    if numtoks < max_length and pad and tokenizer.pad_token_id is not None:
        padlen = max_length - numtoks
        padlist = [tokenizer.pad_token_id] * padlen
        ids = ids + padlist
        no_attention = [0] * padlen
        attention_mask = attention_mask + no_attention
    return ids, attention_mask


def manual_batch_encode(instrlist, tokenizer, truncation=True, max_length=512, 
                        pad=True, indiv_digits=True):
    """ Manually encode a list of strings for T5, BART, Roberta or GPT2
    Returns dict {'input_ids': [ [], [] ],
                  'attention_mask': [ [], [] ] }
    """
    outdict = {}
    input_ids_list = []
    attention_mask_list = []
    for instr in instrlist:
        input_ids, attention_mask = manual_encode(instr, tokenizer, 
                                                  truncation=truncation,
                                                  max_length=max_length, 
                                                  pad=pad, 
                                                  indiv_digits=indiv_digits)
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
    outdict['input_ids'] = input_ids_list
    outdict['attention_mask'] = attention_mask_list
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
        self.data = []
        with open(self.data_path, "r") as f:
            cnt = 0
            invalid_lines = 0
            for line in f:
                try:
                    question, answer = line.split("\t")
                except Exception:
                    invalid_lines += 1
                    continue
                self.data.append({
                    "id": "{}-{}".format(self.data_type, cnt),
                    "question": question,
                    "answer": [answer]
                })
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
        new_answers, metadata = [], []
        for answer in answers:
            assert type(answer)==list
            metadata.append((len(new_answers), len(new_answers)+len(answer)))
            new_answers += answer
        return new_answers, metadata

    def load_dataset(self, tokenizer, do_return=False):
        self.tokenizer = tokenizer
        postfix = tokenizer.__class__.__name__.replace("zer", "zed")
        preprocessed_path = os.path.join(
            "/".join(self.data_path.split("/")[:-1]),
            self.data_path.split("/")[-1].replace(
                ".tsv" if self.data_path.endswith(".tsv") else ".json",
                "{}{}{}{}{}{}-{}.json".format(
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
                    metadata = json.load(f)
        else:
            print ("Start tokenizing...")
            manually_add_special_tokens = False
            dopad = True
            if self.tokenizer.pad_token_id is None:   # gpt2 doesnt have a pad token
                dopad = False
                self.logger.info("Not padding since tokenizer has no padding token.")
                manually_add_special_tokens = True  # GPT2 doesnt add special tokens even when add_special_tokens =True in batch_encode_plus
                
            if self.args.append_another_bos and self.tokenizer.bos_token_id is None:
                self.logger.info("Tokenizer has no bos token so ignoring --append_another_bos flag.")
                bos_token = ''  # T5 doesnt have BOS token
            else:
                bos_token = self.tokenizer.bos_token + ' '  # gpt2 bos token = eos token...
            
            questions = [d["question"] if d["question"].endswith("?") else d["question"]+"?"
                        for d in self.data]
            answers = [d["answer"] for d in self.data]
            answers, metadata = self.flatten(answers)

            if self.args.norm_numbers:
                norm = ''
                if self.args.norm_10e:
                    norm = '10e'
                questions = normalize_num_batch(questions, norm=norm)
                answers = normalize_num_batch(answers, norm=norm)

            if self.args.strip_single_quotes:   # Added for T5
                questions = [re.sub("'(.*)'", r"\1", question) for question in questions]
                answers = [re.sub("'(.*)'", r"\1", answer) for answer in answers]
                
            if self.args.do_lowercase:
                questions = [question.lower() for question in questions]
                answers = [answer.lower() for answer in answers]

            if self.args.append_another_bos:
                questions = [bos_token + question for question in questions]
                answers = [bos_token + answer for answer in answers]
                                
            if self.args.indiv_digits:
                question_input = manual_batch_encode(questions, 
                                                     self.tokenizer,
                                                     truncation=True,
                                                     pad=dopad,
                                                     max_length=self.args.max_input_length,
                                                     indiv_digits=self.args.indiv_digits)
                answer_input = manual_batch_encode(answers, 
                                                     self.tokenizer,
                                                     truncation=True,
                                                     pad=dopad,
                                                     max_length=self.args.max_output_length,
                                                     indiv_digits=self.args.indiv_digits)
            elif dopad:  
                    question_input = self.tokenizer.batch_encode_plus(questions,
                                                                      truncation=True,       
                                                                      padding='max_length',  
                                                                      max_length=self.args.max_input_length)
                    answer_input = self.tokenizer.batch_encode_plus(answers,
                                                                    truncation=True,       
                                                                    padding='max_length',  
                                                                    max_length=self.args.max_output_length)
            else:  # dont pad
                    question_input = self.tokenizer.batch_encode_plus(questions,
                                                                      truncation=True,       
                                                                      max_length=self.args.max_input_length)
                    answer_input = self.tokenizer.batch_encode_plus(answers,
                                                                    truncation=True,       
                                                                    max_length=self.args.max_output_length)
                
            input_ids, attention_mask = question_input["input_ids"], question_input["attention_mask"]
            decoder_input_ids, decoder_attention_mask = answer_input["input_ids"], answer_input["attention_mask"]
            
            if self.load:
                preprocessed_data = [input_ids, attention_mask,
                                     decoder_input_ids, decoder_attention_mask,
                                     metadata]
                with open(preprocessed_path, "w") as f:
                    json.dump([input_ids, attention_mask,
                               decoder_input_ids, decoder_attention_mask,
                               metadata], f)
                self.logger.info("Saved tokenised data to {}".format(preprocessed_path))

        self.dataset = MyQADataset(input_ids, attention_mask,
                                         decoder_input_ids, decoder_attention_mask,
                                         in_metadata=None, out_metadata=metadata,
                                         is_training=self.is_training)
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
            

def get_exact_match(prediction, groundtruth):
    if type(groundtruth)==list:
        if len(groundtruth)==0:
            return 0
        return np.max([get_exact_match(prediction, gt) for gt in groundtruth])
    return (normalize_answer(prediction) == normalize_answer(groundtruth))


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


class MyQADataset(Dataset):
    def __init__(self,
                 input_ids, attention_mask,
                 decoder_input_ids, decoder_attention_mask,
                 in_metadata=None, out_metadata=None,
                 is_training=False):
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_mask = torch.LongTensor(attention_mask)
        self.decoder_input_ids = torch.LongTensor(decoder_input_ids)
        self.decoder_attention_mask = torch.LongTensor(decoder_attention_mask)
        self.in_metadata = list(zip(range(len(input_ids)), range(1, 1+len(input_ids)))) \
            if in_metadata is None else in_metadata
        self.out_metadata = list(zip(range(len(decoder_input_ids)), range(1, 1+len(decoder_input_ids)))) \
            if out_metadata is None else out_metadata
        self.is_training = is_training

        assert len(self.input_ids)==len(self.attention_mask)==self.in_metadata[-1][-1]
        assert len(self.decoder_input_ids)==len(self.decoder_attention_mask)==self.out_metadata[-1][-1]

    def __len__(self):
        return len(self.in_metadata)

    def __getitem__(self, idx):
        if not self.is_training:
            idx = self.in_metadata[idx][0]
            return self.input_ids[idx], self.attention_mask[idx]

        in_idx = np.random.choice(range(*self.in_metadata[idx]))
        out_idx = np.random.choice(range(*self.out_metadata[idx]))
        return self.input_ids[in_idx], self.attention_mask[in_idx], \
            self.decoder_input_ids[out_idx], self.decoder_attention_mask[out_idx]


class MyDataLoader(DataLoader):

    def __init__(self, args, dataset, is_training):
        if is_training:
            sampler=RandomSampler(dataset)
            batch_size = args.train_batch_size
        else:
            sampler=SequentialSampler(dataset)
            batch_size = args.predict_batch_size
        super(MyDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size)


