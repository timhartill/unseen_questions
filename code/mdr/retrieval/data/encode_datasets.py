# Portions Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
"""
Encode a corpus and save embeddings as index.npy [#paras, hidden_size] 
and associated text as id2doc.json {index.npy para idx : [unescape(title), text [, para_idx = wikidocid_paraidx] ]}
"""

import csv
import json
import pdb
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import codecs
from .data_utils import collate_tokens
import unicodedata
import re
import os
from html import unescape

from utils import encode_text


def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)

def convert_brc(string):
    string = re.sub('-LRB-', '(', string)
    string = re.sub('-RRB-', ')', string)
    string = re.sub('-LSB-', '[', string)
    string = re.sub('-RSB-', ']', string)
    string = re.sub('-LCB-', '{', string)
    string = re.sub('-RCB-', '}', string)
    string = re.sub('-COLON-', ':', string)
    return string

class EmDataset(Dataset):

    def __init__(self,
                 tokenizer,
                 data_path,
                 max_q_len,
                 max_c_len,
                 is_query_embed,
                 save_path
                 ):
        super().__init__()
        self.is_query_embed = is_query_embed
        self.tokenizer = tokenizer
        self.max_c_len = max_c_len

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_path = os.path.join(save_path, "id2doc.json") # ID to doc mapping

        print(f"Loading data from {data_path}")
        self.data_format = 'abstracts'

        if self.is_query_embed:
            self.data = [json.loads(_.strip())
                        for _ in tqdm(open(data_path).readlines())]
        else:
            if data_path.endswith("tsv"):
                self.data = []
                with open(data_path) as tsvfile:
                    reader = csv.reader(tsvfile, delimiter='\t', )
                    for row in reader:
                        if row[0] != 'id':
                            id_, text, title = row[0], row[1], row[2]
                            self.data.append({"id": id_, "text": text, "title": title})
            elif "fever" in data_path:
                raw_data = [json.loads(l) for l in tqdm(open(data_path).readlines())]
                self.data = []
                for _ in raw_data:
                #     _["title"] = normalize(_["title"])
                    # _["title"] = convert_brc(_["title"])
                    # _["text"] = convert_brc(_["text"])
                    self.data.append(_)                              
            else:  #hpqa or beerqa path
                # hpqa format: {"title": "One Night Stand (1984 film)", "text": "One Night Stand is a 1984 film directed by John Duigan."}
                # beerqa format: see convert_beerqa_wikipedia_paras.py docstring
                self.data = [json.loads(l) for l in open(data_path).readlines()]  # Note takes ~30 mins for full wikipedia load
                if self.data[0].get('paras') is not None:
                    self.data_format = 'paras'
                
            print(f"load {len(self.data)} documents with '{self.data_format}' format...")
            id2doc = {}
            if self.data_format == 'abstracts':
                for idx, doc in enumerate(self.data):
                    id2doc[idx] = (unescape(doc["title"]), doc["text"])  #TJH removed, doc.get("intro", False)) Also unescaping title 
            else:
                new_data = []
                idx = 0
                for doc in self.data:
                    for para_idx, para in enumerate(doc['paras']):
                        newid = doc['id'] + '_' + str(para_idx)
                        title_unescaped = unescape(doc["title"]) # use unescaped title
                        id2doc[idx] = (title_unescaped, para["text"], newid)  # idx is numeric here but when saved to json it's a str..
                        new_data.append( {"title": title_unescaped, "text": para["text"]} )  #don't need para_id for __getitem__()
                        idx += 1
                self.data = new_data
                        
            print(f"Saving {len(id2doc)} paras into {save_path}...")
            with open(save_path, "w") as g:
                json.dump(id2doc, g)  #tuple saved as list..

        self.max_len = max_q_len if is_query_embed else max_c_len
        print(f"Max sequence length: {self.max_len}")


    def __getitem__(self, index):
        sample = self.data[index]

        if "Roberta" in self.tokenizer.__class__.__name__ and sample["text"].strip() == "":
            print(f"empty doc title: {sample['title']}")
            sample["text"] = sample["title"]
        # if sample["text"].endswith("."):
        #     sample["text"] = sample["text"][:-1]

        sent_codes = encode_text(self.tokenizer, normalize(sample["title"].strip()), text_pair=sample['text'].strip(), max_input_length=self.max_len, truncation=True, padding=False, return_tensors="pt")
        #sent_codes = self.tokenizer.encode_plus(normalize(sample["title"].strip()), text_pair=sample['text'].strip(), max_length=self.max_len, truncation=True, return_tensors="pt")
        return sent_codes
    

    def __len__(self):
        return len(self.data)


def em_collate(samples):
    if len(samples) == 0:
        return {}

    batch = {
        'input_ids': collate_tokens([s['input_ids'].view(-1) for s in samples], 0),
        'input_mask': collate_tokens([s['attention_mask'].view(-1) for s in samples], 0),
    }

    if "token_type_ids" in samples[0]:
        batch["input_type_ids"] = collate_tokens([s['token_type_ids'].view(-1) for s in samples], 0)

    return batch
