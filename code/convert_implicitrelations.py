#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 15:57:27 2022

Convert ImplicitRelations dev/test samples into UQA format 
and rebalance s.t. train = orig dev + 0.5 of test (150 samples * 3 = 750) and dev = remaining 0.5 of test (50 samples * 3 = 150 samples)  

Each sample create a separate output sample for each combo of entity:relation

@author: tim hartill

"""
import os
import json
import random

import eval_metrics
import utils

fsqa_dev = '/home/thar011/data/implicitrelations/ImplicitRelations/StrategyQA_ImplicitRelations_dev.json'
fsqa_test = '/home/thar011/data/implicitrelations/ImplicitRelations/StrategyQA_ImplicitRelations_test.json'

fcreak_dev = '/home/thar011/data/implicitrelations/ImplicitRelations/creak_ImplicitRelations_dev.json'
fcreak_test = '/home/thar011/data/implicitrelations/ImplicitRelations/creak_ImplicitRelations_test.json'

fcsqa2_dev = '/home/thar011/data/implicitrelations/ImplicitRelations/CommonsenseQA 2_ImplicitRelations_dev.json'
fcsqa2_test = '/home/thar011/data/implicitrelations/ImplicitRelations/CommonsenseQA 2_ImplicitRelations_test.json'

UQA_DIR = eval_metrics.UQA_DIR

sqa_dev = json.load(open(fsqa_dev))
sqa_test = json.load(open(fsqa_test))

creak_dev = json.load(open(fcreak_dev))
creak_test = json.load(open(fcreak_test))

csqa2_dev = json.load(open(fcsqa2_dev))
csqa2_test = json.load(open(fcsqa2_test))


#TODO - must identify whether any of the SQA samples are in the new SQA dev split. Apparently all of them came from the SQA orig train split
#TODO - with or without a context?
#       q -> <concept1>:keyword1; <concept2>:keyword2
# or    q + c -> <concept1>:keyword1; <concept2>:keyword2   what is c? sqa, creak: explanation but csqa2 no explanation and sqa test no explanation
# dev set answers should be list of all combos in difft orders so "wrong" order counts. # elements in list = 2 x #concept1 x #concept2 x ...
# train set should have separate sample per combo

def cleanup(sample):
    """  [[['Jane Austen', 'position among siblings']],
          [['Jane Austen', 'Birth order']],
          [['middle child syndrome', 'causes'], ['Jane Austen', 'family birth order']]] 
    """
    flattened_pairs = utils.flatten(sample['implicit_relation_pairs'])  # [concept, relation, concept2, relation, concept, relation, ...]
    assert len(flattened_pairs) % 2 == 0
    pairs = list(set([(flattened_pairs[i].strip().lower(), flattened_pairs[i+1].strip().lower()) for i in range(0, len(flattened_pairs),2)]))
    combos = []
    for i, pair in enumerate(pairs):  # combine into [ [(c1,r1), (c2,r2)], [(c1,r1), (c3,r3)], ... ]
        for j, pair2 in enumerate(pairs):
            if pair2[0] != pair[0] and j > i:
                combos.append( [pair, pair2] )
    if combos == []:  # only one concept is annotated
        for i, pair in enumerate(pairs):  # combine into [ [(c1,r1)], [(c1,r2)], ... ]
            combos.append([pair])
    sample['combos'] = combos
    return


def process(splits):
    for split in splits:
        for sample in split:
            cleanup(sample)
    return


def output_train_dev(train, dev):
    out_train = []
    for sample in train:
        for combo in sample['combos']:
            answer = ''
            for pair in combo:
                answer += pair[0] + ': ' + pair[1] + '; '
            #print(answer)
            out_train.append( utils.create_uqa_example(sample['question'], ' ', answer.strip()) )
            
            # answer with order reversed
            if len(combo) > 1:
                answer = ''
                for pair in combo[::-1]:
                    answer += pair[0] + ': ' + pair[1] + '; '
                #print(answer)
                out_train.append( utils.create_uqa_example(sample['question'], ' ', answer.strip()) )
    
    out_dev = []
    for sample in dev:
        answer_list = []
        for combo in sample['combos']:
            answer = ''
            for pair in combo:
                answer += pair[0] + ': ' + pair[1] + '; '
            answer_list.append(answer.strip())
            if len(combo) > 1:
                answer = ''
                for pair in combo[::-1]:
                    answer += pair[0] + ': ' + pair[1] + '; '
                answer_list.append(answer.strip())
        print(answer_list)
        out_dev.append( utils.create_uqa_example(sample['question'], ' ', answer_list) )
        
    out_dir = os.path.join(UQA_DIR, "implicit_relations")
    print(f'Outputting to {out_dir}')
    os.makedirs(out_dir, exist_ok=True)
    outfile = os.path.join(out_dir, 'dev.tsv')
    print(f"Outputting: {outfile}")
    with open(outfile, 'w') as f:
        f.write(''.join(out_dev))
        
    outfile = os.path.join(out_dir, 'train.tsv')
    print(f"Outputting: {outfile}")
    with open(outfile, 'w') as f:
        f.write(''.join(out_train))
    print('Finished outputting implicit_relations!')
    return
        

process([sqa_dev, sqa_test, creak_dev, creak_test, csqa2_dev, csqa2_test])

random.seed(42)
random.shuffle(sqa_test)
random.shuffle(creak_test)
random.shuffle(csqa2_test)

train = sqa_dev + sqa_test[:50] + creak_dev + creak_test[:50] + csqa2_dev + csqa2_test[:50] #465
dev = sqa_test[50:] + creak_test[50:] + csqa2_test[50:]  #150

output_train_dev(train, dev)

