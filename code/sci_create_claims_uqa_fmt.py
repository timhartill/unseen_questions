#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 10:06:02 2022

@author: thar011
"""
import pandas as pd
import copy
import utils
from text_processing import create_sentence_spans, split_into_sentences
import eval_metrics


claimfile = '/home/thar011/data/SCI/claims.txt'
questions = utils.loadas_txt(claimfile)
outfile = '/data/thar011/data/unifiedqa/claims_test/test.tsv'
utils.create_uqa_from_list(questions, outfile, answers=None, ans_default='NO ANS PROVIDED')


scicorpusfile = '/home/thar011/data/SCI/corpus.jsonl'
scicorpusfileout = '/home/thar011/data/SCI/sci_corpus_with_sent_spans.jsonl'

sci_corpus = utils.load_jsonl(scicorpusfile)  #dict_keys(['doc_id', 'title', 'abstract', 'metadata', 'scifact_orig'])

#v = sci_corpus[0]
#create_sentence_spans(v['abstract'])
#' '.join(v['abstract'])


sci_abstracts_out = [{'title': v['title'], 'text': ' '.join(v['abstract']), 
                      'sentence_spans': create_sentence_spans(v['abstract'])} for v in sci_corpus] # 500,000
utils.saveas_jsonl(sci_abstracts_out, scicorpusfileout)

#sci_abstracts_out[0]

scicorpusfileout_paras = '/home/thar011/data/SCI/sci_corpus_paras_with_sent_spans.jsonl'

max_sents = 5

sci_corpus_paras_out = []
for sample in sci_corpus:
    outsents = []
    i = 0
    for sent in sample['abstract']:
        if i < max_sents:
            outsents.append(sent)
        else:
            i = 0
            text = ' '.join(outsents)
            spans = create_sentence_spans(outsents)
            sci_corpus_paras_out.append({'title': sample['title'], 'text': text, 'sentence_spans': spans})
            outsents = []
        i += 1
    if len(outsents) > 0:
        text = ' '.join(outsents)
        spans = create_sentence_spans(outsents)
        sci_corpus_paras_out.append({'title': sample['title'], 'text': text, 'sentence_spans': spans})
        
print(f"# chunks: {len(sci_corpus_paras_out)}")       ## chunks: 929650
utils.saveas_jsonl(sci_corpus_paras_out, scicorpusfileout_paras)


############################################
# Evaluate retrieval
###############################################
def get_best_hop(sample):
    """ for s2 append the final hop to the historical hops and choose the best one based on highest s2ev_score
    If run to max_hops (eg ev score thresh higher than max ev score encountered) it's possible for an intermediate hop
    to actually be the best one.
    sample['s2_pred_hist']: list of [sample['s2_ans_pred'], sample['s2_ans_pred_score'], sample['s2_ans_insuff_score'], sample['s2_ans_conf_delta'], sample['s2_ev_score']]
    """
    sample['s2_hist_all'] = copy.deepcopy( sample['s2_hist'] )
    if sample.get('s2_full') is not None:
        sample['s2_hist_all'].append( sample['s2_full'] )  # current s2 is final hop so put it on end of hist list
    else:  # backwards compatability
        sample['s2_hist_all'].append( sample['s2'] )
    sample['s2_pred_hist_all'] = copy.deepcopy( sample['s2_pred_hist'] )
    sample['s2_pred_hist_all'].append( [ sample['s2_ans_pred'], sample['s2_ans_pred_score'], sample['s2_ans_insuff_score'], sample['s2_ans_conf_delta'], sample['s2ev_score'] ] )
    best_score = -1.0
    best_hop = -1
    for i, pred_hist in enumerate(sample['s2_pred_hist_all']):
        if pred_hist[4] > best_score:
            best_score = pred_hist[4]
            best_hop = i
            sample['best_hop'] = best_hop + 1
            sample['total_hops'] = len(sample['s2_hist_all'])
            sample['s2_best'] = sample['s2_hist_all'][best_hop]
            for s in sample['s2_best']:
                s['s2ev_score'] = best_score  # early bug had s2_hist ev scores all set to the latest one so copy the true s2_evscore back from pred_hist
            sample['s2_best_preds'] = {'s2_ans_pred': pred_hist[0], 's2_ans_pred_score':pred_hist[1], 
                                       's2_ans_insuff_score': pred_hist[2], 's2_ans_conf_delta': pred_hist[3],
                                       's2ev_score': pred_hist[4], 's2_best_hop': best_hop}
    return

def join_annots(samples):
    found = 0
    not_found = []
    for s in samples:
        get_best_hop(s)
        c = s['question']
        if c[-1] == '.':
            c = c[:-1]
        if c.startswith("BRCA 1 mutation carriers' risk of breast and ovarian cancer is not"):
            c = "BRCA 1 mutation carriers' risk of breast and ovarian cancer  is not influence by the location of the mutation"
        if annots.get(c):
            s['annots'] = annots[c]
            found += 1
        else:
            not_found.append(c)
    print(f"found:{found}")
    print(not_found)

def print_titles(samples):
    all_f1 = []
    for s in samples:
        if s.get('annots'):
            print('QUESTION:', {s['question']})
            print('GOLD:', [t['title'] for t in s['annots']])
            print('RETRIEVED:', [t['title'] for t in s['s2_best']])
            f1 = eval_metrics.get_f1([t['title'] for t in s['s2_best']], [t['title'] for t in s['annots']])
            all_f1.append(f1)
            print(f"F1: {f1}")
            print()
    print(f"MEAN F1: {sum(all_f1)/len(all_f1)}")
    return


def print_retrieved(samples):
    for s in samples:
        if s.get('annots'):
            print('QUESTION:', {s['question']})
            print('RETRIEVED NOT GT:', [(t['title'], t['sentence']) for t in s['s2_best'] if t['title'] not in [x['title'] for x in s['annots']] ])
            print('RETRIEVED GT:', [(t['title'], t['sentence']) for t in s['s2_best'] if t['title'] in [x['title'] for x in s['annots']] ])
    return    

annots14file = '/home/thar011/data/SCI/gold_annots_14_claims.csv'
abstractsfull = '/large_data/thar011/out/mdr/logs/SCI_ITER_scifullabstracts_claimstest_test2-12-14-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue/samples_with_context.jsonl'
abstractsparas = '/large_data/thar011/out/mdr/logs/SCI_ITER_sciparas_claimstest_test3-12-14-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue/samples_with_context.jsonl'

df = pd.read_csv(annots14file)  # ['Unnamed: 0', 'claim_id', 'claim', 'label', 'title', 'abstract', 'rationales', 'abstract_id']
annots = {}
for r in df.itertuples():
    c = r.claim
    if c[-1] == '.':
        c = c[:-1]
    if annots.get(c) is None:
        annots[c] = []
    annots[c].append( {'label': r.label, 'title': r.title, 'abstract': r.abstract, 'gold_sents': r.rationales, 'abstract_id': r.abstract_id} )    
    


full = utils.load_jsonl(abstractsfull)   # list of {}: dict_keys(['question', 'answer', 'mc_options', 'init_context', 'src', 'type', '_id', 'dense_retrieved', 's1', 's2', 's2_full', 's2_ans_pred', 's2_ans_pred_score', 's2_ans_insuff_score', 's2_ans_conf_delta', 's2ev_score', 'stop_reason', 'dense_retrieved_hist', 's1_hist', 's2_hist', 's2_pred_hist'])
paras = utils.load_jsonl(abstractsparas)

join_annots(full) # dict_keys(['question', 'answer', 'mc_options', 'init_context', 'src', 'type', '_id', 'dense_retrieved', 's1', 's2', 's2_full', 's2_ans_pred', 's2_ans_pred_score', 's2_ans_insuff_score', 's2_ans_conf_delta', 's2ev_score', 'stop_reason', 'dense_retrieved_hist', 's1_hist', 's2_hist', 's2_pred_hist', 's2_hist_all', 's2_pred_hist_all', 'best_hop', 'total_hops', 's2_best', 's2_best_preds', 'annots'])
join_annots(paras)

print_titles(full)  # MEAN F1: 0.08714285714285716
print_titles(paras) # MEAN F1: 0.12714285714285714

print_retrieved(paras)

