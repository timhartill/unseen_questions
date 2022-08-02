""" Multi-dataset Loader

Author: Tim Hartill

Portions Adapted from https://github.com/allenai/unifiedqa
"""

import os
import json
import re
import string
import time
import numpy as np

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

#from transformers import AutoTokenizer

from data import QAData, MyDataLoader, manual_batch_encode, normalize_num_batch, selfsupervisedkey, pad_list, self_supervise

import utils
from eval_metrics import get_exact_match, parse_mixture


def sample_datasets(logger, data, approx_dev_samples):
    """ Restrict size of each dev dataset to ~approx_dev_samples for reasonable validation step time
    """
    logger.info(f"Loaded data for dev. Now restricting each dev dataset to approximately {approx_dev_samples} samples.")
    sampled_data = {}
    for dataset in data.keys():
        datasetlen = len(data[dataset]["question"])
        n_skip = round(datasetlen / approx_dev_samples)
        if n_skip <= 1:   
            sampled_data[dataset] = data[dataset]
        else:
            sampled_data[dataset] = {"id": [], "question": [], "answer": []}
            for i in range(0, datasetlen, n_skip):
                 sampled_data[dataset]["id"].append( data[dataset]["id"][i] )
                 sampled_data[dataset]["question"].append( data[dataset]["question"][i] )
                 sampled_data[dataset]["answer"].append( data[dataset]["answer"][i] )
    for dataset in sampled_data.keys():
        logger.info(f"{dataset}: New count:{len(sampled_data[dataset]['question'])} Orig:{len(data[dataset]['question'])}")
    return sampled_data


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

        self.selfsupervised = []
        self.data = {}
        for dataset in self.unified_dataset:
            if selfsupervisedkey in dataset:
                self.selfsupervised.append(True)
            else:
                self.selfsupervised.append(False)
            
            assert data_path.endswith(".tsv"), "data file has to be in tsv format"
            curr_data_path = data_path.replace(f"{self.data_type}.tsv", f"{dataset}/{self.data_type}.tsv")
            print(f"Dataset: Loading {curr_data_path}..")
            self.data[dataset] = {"id": [], "question": [], "answer": []}
            with open(curr_data_path, "r") as f:
                cnt = 0
                for line in f:
                    if self.selfsupervised[-1]:
                        answer = ""
                        question = line.strip()
                    else:
                        question, answer = line.split("\t")
                        
                        #always train with single answers but can eval EM/F1 with multiple answers
                        if answer.lstrip().startswith(utils.MULTI_ANS_SEP): # #!# answer 1#!# answer 2 #!# -> ['answer 1','answer 2']
                            answer = answer.strip().strip(utils.MULTI_ANS_SEP).split(utils.MULTI_ANS_SEP)
                            answer = [a + '\n' for a in answer]
                        else:
                            answer = answer # note always ends in \n  and if singular is str not list unlike data.py                   

                    self.data[dataset]["id"].append("{}-{}-{}".format(dataset, self.data_type, cnt))
                    self.data[dataset]["question"].append(question)
                    self.data[dataset]["answer"].append(answer)
                    cnt += 1
                    if args.debug and cnt==20:
                        break

        if not is_training and args.approx_dev_samples != -1:  # limit number of dev samples so validation step takes reasonable amount of time
            self.data = sample_datasets(logger, self.data, args.approx_dev_samples)  
        if is_training:
            logger.info("Loaded data for training:")
        else:
            logger.info("Loaded data for dev/test:")
        for dataset in self.data.keys():
            logger.info(f"{dataset}: count:{len(self.data[dataset]['question'])} samples")
                  
                    
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

    def load_dataset(self, tokenizer, load_preprocessed=True):
        if not load_preprocessed: 
            self.load = False  # don't load or save tokenised data to file
        self.tokenizer = tokenizer
        postfix = self.tokenizer.__class__.__name__.replace("zer", "zed")
        preprocessed_path = os.path.join(
                "/".join(self.data_path.split("/")[:-1]),
                self.data_path.split("/")[-1].replace(".tsv", "-v3-{}{}{}{}{}{}-{}-{}.json".format(
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
                input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, metadata, word_starts, ners_ids = json.load(f)
        else:
            self.logger.info("Calculating metadata...")            
            metadata, questions, answers = [], [], []
            for dataset in self.unified_dataset:  # metadata = [(start idx ds1, end idx ds1), (start ds2, end ds2),...]
                metadata.append((len(questions), len(questions)+len(self.data[dataset]["question"])))  # store start & end indices of each dataset in metadata
                questions += self.data[dataset]["question"]  # final questions = single list concatenating questions from each constituent ds
                answers += [ d if type(d) == str else d[0] for d in self.data[dataset]["answer"] ]  #only tokenise 1st answer if list

            if self.args.dont_pretokenize:
                self.logger.info("Not pre-tokenizing.")
                word_starts, ners_ids, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask = [], [], [], [], [], []
            else:                
                self.logger.info("Pre-tokenizing questions...")
                question_input = manual_batch_encode(questions, 
                                                     self.tokenizer,
                                                     self.logger,
                                                     self.args,
                                                     self.selfsupervised,
                                                     metadata,
                                                     truncation=True,
                                                     pad=False,
                                                     max_length=self.args.max_input_length)
                self.logger.info("Pre-tokenizing answers...")
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
                print ("Finished tokenizing...")
                if self.load:
                    with open(preprocessed_path, "w") as f:
                        json.dump([input_ids, attention_mask,
                                   decoder_input_ids, decoder_attention_mask, metadata, word_starts, ners_ids], f)
                    self.logger.info("Saved tokenised data to {}".format(preprocessed_path))

        self.metadata = metadata
        self.err_sampler = SampleProbs(self.unified_dataset, self.selfsupervised, self.args.error_based_ssvise_prob)
        if self.is_training and self.args.error_based_sampling:
            self.logger.info(f"Initial err based sampling probs: {self.err_sampler.current_probs_string()}")
        self.dataset = MyUnifiedQADataset(input_ids, attention_mask,
                                          decoder_input_ids, decoder_attention_mask, self.args, self.data,
                                          metadata=metadata, is_training=self.is_training,
                                          tokenizer=self.tokenizer,
                                          selfsupervised = self.selfsupervised,
                                          word_starts = word_starts,
                                          ners_ids=ners_ids, err_sampler=self.err_sampler)
        self.logger.info("Loaded {} examples from {} data".format(len(self.dataset), self.data_type))



    def load_dataloader(self, do_return=False):
        self.dataloader = MyDataLoader(self.args, self.dataset, self.is_training, self.tokenizer.pad_token_id)
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


def build_objective_indx(metadata, selfsupervised):
    """ Return a list of true/false indicating whether objective[idx] is self supervised or not
    """
    objective = []
    for i,(start, end) in enumerate(metadata):
        num_to_add = end - start
        objective += [selfsupervised[i]] * num_to_add
    return objective 


def get_parentdata_indx(idx, metadata, unified_dataset):
    """ Given an index to input_ids, return the corresponding dataset name and index
        into that dataset.
    """
    for i,(start, end) in enumerate(metadata):  
        if idx >= start and idx < end:
            dset = unified_dataset[i]
            ds_idx = idx - start
            break
    return dset, ds_idx


class SampleProbs():
    """ Update task sampling probs based on error sampling
    """
    def __init__(self, unified_datasets, selfsupervised, selfsupervised_prob=0.5, sampletype='err'):
        """ selfsupervised = [True, False, ...]  whether each dataset is self-supervised or not
        Return idx of a self-supervised dataset with prob selfsupervised_prob else return idx of a "normal" dataset
        Within self-supervised and normal sets return idx of dataset with probability based on error based sampling ie oversample dataset with higher error rate (1 - accuracy)
        """
        self.unified_datasets = unified_datasets
        self.selfsupervised = selfsupervised
        self.num_datasets = len(selfsupervised)
        self.dataset_idx = [i for i in range(self.num_datasets)] # the actual dataset idx ssvise+normal
        self.ssvise_prob = selfsupervised_prob
        self.ssvise_idxs = []
        self.normal_idxs = []
        for i in self.dataset_idx:
            if self.selfsupervised[i]:
                self.ssvise_idxs.append(i)
            else:
                self.normal_idxs.append(i)
        self.num_ssvise = len(self.ssvise_idxs)
        self.num_normal = len(self.normal_idxs)
        
        self.sampleprobs_ssvise = np.array([1.0/self.num_ssvise for _ in range(self.num_ssvise)])
        self.sampleprobs_norm = np.array([1.0/self.num_normal for _ in range(self.num_normal)])
        self.sampletype = sampletype
        
    def update(self, ems, verbose=False):
        if self.sampletype=='err':
            acctotal_norm = np.sum([1.0-acc for i, acc in enumerate(ems) if not self.selfsupervised[i]])
            acctotal_ssvise = np.sum([1.0-acc for i, acc in enumerate(ems) if self.selfsupervised[i]])
            for i, acc in enumerate(ems):
                if self.selfsupervised[i]:
                    for nidx, j in enumerate(self.ssvise_idxs):
                        if j==i:
                            self.sampleprobs_ssvise[nidx] = (1.0-acc) / acctotal_ssvise
                            break
                else:
                    for nidx, j in enumerate(self.normal_idxs):
                        if j==i:
                            self.sampleprobs_norm[nidx] = (1.0-acc) / acctotal_norm
                            break
        if verbose:
            print(f"New probs: {self.current_probs_string()}")
        return
    
    def sample(self):
        group_choice = np.random.choice(['ssvise', 'norm'], p=[self.ssvise_prob, 1.0-self.ssvise_prob])
        if group_choice == 'ssvise' and self.num_ssvise == 0:
            group_choice = 'norm'
        elif group_choice == 'norm' and self.num_normal == 0:
            group_choice = 'ssvise'
        if group_choice == 'ssvise':
            return np.random.choice(self.ssvise_idxs, p=self.sampleprobs_ssvise)
        else:
            return np.random.choice(self.normal_idxs, p=self.sampleprobs_norm)
        
    def current_probs_string(self):
        outstr_ssvise = f'SSVISE_DS({self.ssvise_prob}):'
        outstr_norm = 'NORM_DS:'
        for i, name in enumerate(self.unified_datasets):
            if self.selfsupervised[i]:
                outstr_ssvise += ' ' + name + ' '
                for nidx, j in enumerate(self.ssvise_idxs):
                    if j==i:
                        outstr_ssvise += str(round(self.sampleprobs_ssvise[nidx], 3))
                        break
            else:
                outstr_norm += ' ' + name + ' '
                for nidx, j in enumerate(self.normal_idxs):
                   if j==i:
                       outstr_norm += str(round(self.sampleprobs_norm[nidx], 3))
                       break
        return outstr_norm.strip() + ' ' + outstr_ssvise.strip()
    
        


class MyUnifiedQADataset(Dataset):
    def __init__(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, args, data,
                 metadata, is_training=False, tokenizer=None, selfsupervised=None, 
                 word_starts=None, ners_ids=None, err_sampler=None):
        self.args = args
        self.parent_data = data
        self.unified_dataset = list(data.keys())  #same order as unified_dataset in unifiedqadata above..
        self.tokenizer = tokenizer
        if tokenizer is not None and tokenizer.pad_token_id is not None:
            self.pad_token_id = tokenizer.pad_token_id
        else:
            self.pad_token_id = None
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
            self.mask_seq = [m+'_' for m in self.mask_seq]
        if args.add_mask_ctr:
            self.mask_seq = [m+str(i) for i, m in enumerate(self.mask_seq)]
        self.mask_seq = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(m)) for m in self.mask_seq]
        self.eos_token_id = tokenizer.eos_token_id
        self.no_question_label = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("no mask")) 
        self.selfsupervised = selfsupervised                     # [ds1 ssvised t/f, ds2 ssvised t/f, ...]
        self.input_ids = input_ids                              #torch.LongTensor(input_ids)
        self.attention_mask = attention_mask                    #torch.LongTensor(attention_mask)
        self.decoder_input_ids = decoder_input_ids              #torch.LongTensor(decoder_input_ids)
        self.decoder_attention_mask = decoder_attention_mask    #torch.LongTensor(decoder_attention_mask)
        self.metadata = metadata                                # [(ds1 start, ds1 end), (ds2 start, ds2 end), ...]
        self.word_starts = word_starts                          # eg [2, 4, 7,8,...]
        self.ners_ids = ners_ids                                # eg [[[22, 30],[1,9]], [[8, 15]], [[61, 67]]]
        self.is_training = is_training
        self.error_based_sampling = args.error_based_sampling   # # p(task t) = 1.0-acc(t) / sum over all tasks t': (1.0-acc(t'))
        self.err_sampler = err_sampler
        #self.initialize = True  # if numworkers > 0 insufficient to randomly initalize indices in __init__ since dataloadrer makes copies

        assert len(self.input_ids)==len(self.attention_mask)==len(self.decoder_input_ids)==len(self.decoder_attention_mask)==len(self.word_starts)==len(self.ners_ids)
        if not self.args.dont_pretokenize:        
            assert len(self.input_ids)==metadata[-1][-1]
        
        if not self.is_training:
            self.objective = build_objective_indx(self.metadata, self.selfsupervised)

        self.indices = [np.random.permutation(range(start, end)) for start, end in self.metadata]  # list of 11 buckets of component dataset indices
        self.positions = [0 for _ in self.metadata]  # Current position in each bucket. Incremented in __getitem__
        if is_training:
            self.length = len(self.metadata) * np.min([end-start for start, end in self.metadata]) # num of datasets * min num of samples in any dataset
        else: 
            self.length = self.metadata[-1][-1]   #len(self.input_ids)  
        return

    def __len__(self):
        return self.length  #Note 6655 not 391740 if is_training  (# datasets * min # samples in any dataset), if not is_training total # questions

    def __getitem__(self, idx):            
        orig_idx = idx
        if not self.is_training:   # if pretokenised: idx is idx into input_ids etc not idx of component dataset
            ssvise = self.objective[idx] 
            if self.args.dont_pretokenize: # push tokenized input into vars where pretokenised data would have been stored and set idx=0
                dset, ds_idx = get_parentdata_indx(idx, self.metadata, self.unified_dataset)
                question_input = manual_batch_encode([ self.parent_data[dset]['question'][ds_idx] ], 
                                                     self.tokenizer,
                                                     None,
                                                     self.args,
                                                     selfsupervised=[ssvise],
                                                     metadata=[(0,1)],
                                                     truncation=True,
                                                     pad=False,
                                                     max_length=self.args.max_input_length)
                self.input_ids = question_input['input_ids']
                self.attention_mask = question_input['attention_mask']
                self.word_starts = question_input['word_starts']
                self.ners_ids = question_input['ners_ids']
                idx = 0
                
            if ssvise:
                input_ids, attention_mask, decoder_input_ids, decoder_attention_mask = self_supervise(self.args, 
                                                                                                      self.input_ids[idx], 
                                                                                                      self.word_starts[idx], 
                                                                                                      self.ners_ids[idx], 
                                                                                                      self.mask_seq, 
                                                                                                      self.no_question_label,
                                                                                                      self.bos_token_id,
                                                                                                      self.eos_token_id)
                dset, ds_idx = get_parentdata_indx(orig_idx, self.metadata, self.unified_dataset)
                self.parent_data[dset]['answer'][ds_idx] = self.tokenizer.decode(decoder_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            else:
                input_ids, attention_mask = self.input_ids[idx], self.attention_mask[idx]
                #input_ids, attention_mask = pad_list(self.input_ids[idx], self.attention_mask[idx], self.args.max_input_length, self.pad_token_id)
            ###
            #input_ids, attention_mask = pad_list(input_ids, attention_mask, self.args.max_input_length, self.pad_token_id)
            input_ids = torch.LongTensor(input_ids)
            attention_mask = torch.LongTensor(attention_mask)
            return {'input_ids': input_ids, 'attention_mask': attention_mask}

        # Training below..idx becomes index of component dataset
        #if self.initialize:  #didnt work! Seems indices shared across worker processes. Can only use num_workers 0. To use more would need to pick random position each time
        #    np.random.seed(int(str(idx)[-5:]+str(time.time_ns())[-4:]))  # in case seed is carried over into num_workers datasets
        #    self.indices = [np.random.permutation(range(start, end)) for start, end in self.metadata]
        #    self.initialize = False
                
        if self.error_based_sampling:
            idx = self.err_sampler.sample()     # Error based sampling
            #print(self.err_sampler.current_probs_string())
        else:    
            idx = idx % len(self.metadata)      # Select component dataset with uniform chance not proportional to dataset sizes

        ssvise = self.selfsupervised[idx]
        if self.positions[idx]==len(self.indices[idx]):  # If reached the end of this dataset reshuffle dataset indices in bucket
            start, end = self.metadata[idx]
            self.indices[idx] = np.random.permutation(range(start, end))
            self.positions[idx] = 0

        dp_idx = self.indices[idx][self.positions[idx]]  # Select index within input_ids
        dset = self.unified_dataset[idx]
        
        if self.args.dont_pretokenize: # push tokenized input into vars where pretokenised data would have been stored and set dp_idx=0
            question_input = manual_batch_encode([ self.parent_data[dset]['question'][self.positions[idx]] ], 
                                                 self.tokenizer,
                                                 None,
                                                 self.args,
                                                 selfsupervised=[ssvise],
                                                 metadata=[(0,1)],
                                                 truncation=True,
                                                 pad=False,
                                                 max_length=self.args.max_input_length)
            self.input_ids = question_input['input_ids']
            self.attention_mask = question_input['attention_mask']
            self.word_starts = question_input['word_starts']
            self.ners_ids = question_input['ners_ids']
            answer = self.parent_data[dset]['answer'][self.positions[idx]] if type(self.parent_data[dset]['answer'][self.positions[idx]]) == str else self.parent_data[dset]['answer'][self.positions[idx]][0]
            answer_input = manual_batch_encode([ answer ],
                                                 self.tokenizer,
                                                 None,
                                                 self.args,
                                                 selfsupervised=[ssvise],
                                                 metadata=[(0,1)],
                                                 truncation=True,
                                                 pad=False,
                                                 max_length=self.args.max_output_length)
            self.decoder_input_ids = answer_input['input_ids']
            self.decoder_attention_mask = answer_input['attention_mask']
            dp_idx = 0   # index within input_ids always 0 if not pretokenised

        self.positions[idx] += 1
        
        if ssvise:
            #input_ids, attention_mask, decoder_input_ids, decoder_attention_mask = self_supervise(dev_data.dataset.args, dev_data.dataset.input_ids[0], dev_data.dataset.word_starts[0], dev_data.dataset.ners_ids[0], dev_data.dataset.mask_seq, dev_data.dataset.no_question_label, dev_data.dataset.bos_token_id,                                                                                               dev_data.dataset.eos_token_id)
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask = self_supervise(self.args, 
                                                                                                  self.input_ids[dp_idx], 
                                                                                                  self.word_starts[dp_idx], 
                                                                                                  self.ners_ids[dp_idx], 
                                                                                                  self.mask_seq, 
                                                                                                  self.no_question_label,
                                                                                                  self.bos_token_id,
                                                                                                  self.eos_token_id)
        else:
            input_ids, attention_mask = self.input_ids[dp_idx], self.attention_mask[dp_idx]
            decoder_input_ids, decoder_attention_mask = self.decoder_input_ids[dp_idx], self.decoder_attention_mask[dp_idx]

        ###
        #input_ids, attention_mask = pad_list(input_ids, attention_mask, self.args.max_input_length, self.pad_token_id)
        #decoder_input_ids, decoder_attention_mask = pad_list(decoder_input_ids, decoder_attention_mask, self.args.max_output_length, self.pad_token_id)

        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor(attention_mask)
        decoder_input_ids = torch.LongTensor(decoder_input_ids)
        decoder_attention_mask = torch.LongTensor(decoder_attention_mask)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 
                'decoder_input_ids': decoder_input_ids, 'decoder_attention_mask': decoder_attention_mask}

