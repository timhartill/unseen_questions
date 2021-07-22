#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 16:18:23 2021

@author: Tim Hartill

Sentence embeddings using model from:
    Reimers, Nils, and Iryna Gurevych. 2019. 
    “Sentence-BERT: Sentence Embeddings Using Siamese BERT-Networks.” 
    In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 
    3982–92. Hong Kong, China: Association for Computational Linguistics.

"""

import torch
from transformers import AutoTokenizer, AutoModel


class Embedder:
    def __init__(self, cuda=True, model_name = 'sentence-transformers/stsb-roberta-large'):
        self.cuda = cuda
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.get_model()

        
    def mean_pooling(self, model_output, attention_mask):
        """
        Aggregating over token embeddings to get sentence embeddings. See for example:
        https://huggingface.co/sentence-transformers/paraphrase-distilroberta-base-v1
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) # Sum columns
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    
    def get_model(self):
        self.model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.cuda:
            self.model.to(torch.device("cuda"))
        self.model.eval()
        return
    
    
    def encode_inputs(self, sentences):
        """ sentences = ['string of input words 1', 'string of input words 2', ... ]
        """
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, 
                                       max_length=512, return_tensors="pt") 
        if self.cuda:
            encoded_input.to(torch.device("cuda"))
        return encoded_input


    def get_embeddings(self, encoded_input):
        """ Returned sentence embeddings list of num_samples of np shape [1024]
        """
        emb_list = []
        model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, 
                                                encoded_input['attention_mask']).detach().cpu().numpy()
        for emb in sentence_embeddings:
            emb_list.append(emb)
        return emb_list

    def strip_context(self, questions):
        """ Utility to isolate questions from list of "question \\n context.." strings
        """
        new_questions = []
        for q in questions:
            q_split = q.split('\\n')
            if q_split[0].strip() == '':
                new_q = q
            else:
                new_q = q_split[0]
            new_questions.append( new_q )
        return new_questions



