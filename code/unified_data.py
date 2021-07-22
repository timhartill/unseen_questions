""" Multi-dataset Loader

Author: Tim Hartill

Adapted from https://github.com/allenai/unifiedqa
"""

import os
import json
import re
import string
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import AutoTokenizer

from data import QAData, MyDataLoader, manual_batch_encode, normalize_num_batch

def parse_mixture(mixture):
    """ Parse args.mixture and return list of datasets to include plus a key to add 
        to the pretokenised file name.
        args.mixture format: --mixture unifiedqa,extradataset1,extradataset2
    """
    unified_dataset  = []
    mixture_file_key = ''
    mixturelist = mixture.split(',')
    for ds in mixturelist:
        mixture_file_key = mixture_file_key + '_' + ds
        if ds == 'unifiedqa':
            unified_dataset.extend([
            "narrativeqa",
            "ai2_science_middle", "ai2_science_elementary",
            "arc_hard", "arc_easy",
            "mctest_corrected_the_separator",
            "squad1_1", "squad2",
            "boolq",
            "race_string",
            "openbookqa"])
        else:
            unified_dataset.append(ds)
    return unified_dataset, mixture_file_key


class UnifiedQAData(QAData):

    def __init__(self, logger, args, data_path, is_training):
               
        self.unified_dataset, self.mixture_key = parse_mixture(args.mixture)
        self.data_path = data_path  # this would be ../unifiedqa/train.tsv
        self.data_type = data_path.split("/")[-1][:-4]
        assert self.data_type in ["train", "dev", "test"]

        if args.debug:
            self.unified_dataset = self.unified_dataset[:2]
            self.data_type = "dev"
            data_path = data_path.replace("train", "dev")

        self.data = {}
        for dataset in self.unified_dataset:
            assert data_path.endswith(".tsv"), "data file has to be in tsv format"
            curr_data_path = data_path.replace("{}.tsv".format(self.data_type),
                                               "{}/{}.tsv".format(dataset, self.data_type))
            self.data[dataset] = {"id": [], "question": [], "answer": []}
            with open(curr_data_path, "r") as f:
                cnt = 0
                for line in f:
                    question, answer = line.split("\t")
                    self.data[dataset]["id"].append("{}-{}-{}".format(dataset, self.data_type, cnt))
                    self.data[dataset]["question"].append(question)
                    self.data[dataset]["answer"].append(answer)
                    cnt += 1
                    if args.debug and cnt==20:
                        break

        self.is_training = is_training
        self.load = not args.debug
        self.logger = logger
        self.args = args
        self.max_input_length = self.args.max_input_length
        self.tokenizer = None
        self.dataset = None
        self.dataloader = None
        self.cache = None
        self.metric = "Accuracy"
        self.preprocessed_path = None  # set in load_dataset


    def __len__(self):
        return np.sum([len(d["question"]) for d in self.data.values()])

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True).lower()

    def decode_batch(self, tokens):
        return [self.decode(_tokens) for _tokens in tokens]

    def load_dataset(self, tokenizer):
        self.tokenizer = tokenizer
        postfix = self.tokenizer.__class__.__name__.replace("zer", "zed")
        preprocessed_path = os.path.join(
                "/".join(self.data_path.split("/")[:-1]),
                self.data_path.split("/")[-1].replace(".tsv", "{}{}{}{}{}{}-{}-{}.json".format(
                    "-uncased" if self.args.do_lowercase else "",
                    "-xbos" if (self.args.append_another_bos and self.tokenizer.bos_token_id is not None) else "",
                    "-squote" if (self.args.strip_single_quotes) else "",
                    "-idigits" if (self.args.indiv_digits) else "",
                    "-nnorm" if (self.args.norm_numbers) else "",
                    "-10e" if (self.args.norm_10e) else "",
                    postfix, self.mixture_key)))
        self.preprocessed_path = preprocessed_path
        if self.load and os.path.exists(preprocessed_path):
            self.logger.info("Loading pre-tokenized data from {}".format(preprocessed_path))
            with open(preprocessed_path, "r") as f:
                input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, metadata = json.load(f)
        else:
            print ("Start tokenizing...")            
            manually_add_special_tokens = False
            dopad = True
            if self.tokenizer.pad_token_id is None:   # gpt2 doesnt have a pad token                
                dopad = False 
                self.logger.info("Not padding since tokenizer has no padding token.")
                manually_add_special_tokens = True  # TODO GPT2 doesnt add special tokens even when add_special_tokens =True in batch_encode_plus
                
            if self.args.append_another_bos and self.tokenizer.bos_token_id is None:
                self.logger.info("Tokenizer has no bos token so ignoring --append_another_bos flag.")
                bos_token = ''  # T5 doesnt have BOS token
            else:
                bos_token = self.tokenizer.bos_token + ' '  # gpt2 bos token = eos token...
            
                            
            metadata, questions, answers = [], [], []
            for dataset in self.unified_dataset:
                metadata.append((len(questions), len(questions)+len(self.data[dataset]["question"])))  #TJH store start & end indices of each dataset in metadata
                questions += self.data[dataset]["question"]
                answers += self.data[dataset]["answer"]
                
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
            else:
                question_input = self.tokenizer.batch_encode_plus(questions,
                                                                  truncation=True,
                                                                  max_length=self.args.max_input_length)
                answer_input = self.tokenizer.batch_encode_plus(answers,
                                                                truncation=True,
                                                                max_length=self.args.max_output_length)
                
    
            input_ids, attention_mask = question_input["input_ids"], question_input["attention_mask"]
            decoder_input_ids, decoder_attention_mask = answer_input["input_ids"], answer_input["attention_mask"]
            print ("Finish tokenizering...")
            if self.load:
                with open(preprocessed_path, "w") as f:
                    json.dump([input_ids, attention_mask,
                               decoder_input_ids, decoder_attention_mask, metadata], f)
                self.logger.info("Saved tokenised data to {}".format(preprocessed_path))

        self.metadata = metadata
        self.dataset = MyUnifiedQADataset(input_ids, attention_mask,
                                          decoder_input_ids, decoder_attention_mask,
                                          metadata=metadata, is_training=self.is_training)


    def load_dataloader(self, do_return=False):
        self.dataloader = MyDataLoader(self.args, self.dataset, self.is_training)
        if do_return:
            return self.dataloader

    def evaluate(self, predictions):
        assert len(predictions)==len(self), (len(predictions), len(self))
        ems = []
        for i, dataset in enumerate(self.unified_dataset):
            start, end = self.metadata[i]
            _predictions = predictions[start: end]
            assert len(_predictions)==len(self.data[dataset]["answer"])
            em = np.mean([get_exact_match(prediction, gt) for (prediction, gt) \
                          in zip(_predictions, self.data[dataset]["answer"])])
            ems.append(em)
            if self.args.verbose:
                self.logger.info("%s Accuracy = %.2f" % (dataset, 100*em))
        return ems

    def save_predictions(self, predictions):
        assert len(predictions)==len(self), (len(predictions), len(self))
        prediction_dict = {}
        for i, dataset in enumerate(self.unified_dataset):
            start, end = self.metadata[i]
            _predictions = predictions[start: end]
            assert len(_predictions)==len(self.data[dataset]["answer"])
            prediction_dict[dataset] = _predictions
        save_path = os.path.join(self.args.output_dir, "{}validation_predictions.json".format(self.args.prefix))
        with open(save_path, "w") as f:
            json.dump(prediction_dict, f)
        self.logger.info("Saved prediction in {}".format(save_path))

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


class MyUnifiedQADataset(Dataset):
    def __init__(self,
                 input_ids, attention_mask,
                 decoder_input_ids, decoder_attention_mask,
                 metadata,
                 is_training=False):
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_mask = torch.LongTensor(attention_mask)
        self.decoder_input_ids = torch.LongTensor(decoder_input_ids)
        self.decoder_attention_mask = torch.LongTensor(decoder_attention_mask)
        self.metadata = metadata
        self.is_training = is_training

        assert len(self.input_ids)==len(self.attention_mask)==len(self.decoder_input_ids)==len(self.decoder_attention_mask)
        assert len(self.input_ids)==metadata[-1][-1]

        self.indices = [np.random.permutation(range(start, end)) for start, end in self.metadata]  # list of 11 buckets of component dataset indices
        self.positions = [0 for _ in self.metadata]  # Current position in each bucket. Incremented in __getitem__
        self.length = len(self.metadata) * np.min([end-start for start, end in self.metadata]) \
            if is_training else len(self.input_ids)

    def __len__(self):
        return self.length  #Note 6655 not 391740

    def __getitem__(self, idx):
        if not self.is_training:
            return self.input_ids[idx], self.attention_mask[idx]

        idx = idx % len(self.metadata)  # Select component dataset with uniform chance not proportional to dataset sizes
        if self.positions[idx]==len(self.indices[idx]):  # If reached the end of this dataset reshuffle dataset indices in bucket
            start, end = self.metadata[idx]
            self.indices[idx] = np.random.permutation(range(start, end))
            self.positions[idx] = 0

        dp_idx = self.indices[idx][self.positions[idx]]  # Select dataset index within bucket
        self.positions[idx] += 1

        return self.input_ids[dp_idx], self.attention_mask[dp_idx], \
            self.decoder_input_ids[dp_idx], self.decoder_attention_mask[dp_idx]

