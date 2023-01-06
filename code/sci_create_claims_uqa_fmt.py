#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 10:06:02 2022

@author: thar011
"""
import pandas as pd
import copy
import os
import utils
from text_processing import create_sentence_spans, split_into_sentences
import eval_metrics


outfile_base = '/home/thar011/data/SCI/'

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

for s in sci_corpus:
    s['abstract'] = [' '+t if i>0 else t for i, t in enumerate(s['abstract'])]

sci_abstracts_out = [{'title': v['title'], 'text': ''.join(v['abstract']), 
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
            text = ''.join(outsents)
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
def get_best_hop(sample, uptohop = -1):
    """ for s2 append the final hop to the historical hops and choose the best one based on highest s2ev_score
    If run to max_hops (eg ev score thresh higher than max ev score encountered) it's possible for an intermediate hop
    to actually be the best one.
    sample['s2_pred_hist']: list of [sample['s2_ans_pred'], sample['s2_ans_pred_score'], sample['s2_ans_insuff_score'], sample['s2_ans_conf_delta'], sample['s2_ev_score']]
    
    Unlike get_best_hop in mdr_searchers.py this version finds and adds the full para text 
    """
    # set up full para lookup
    ps_ratio = 0.5
    all_retrieved = sample['dense_retrieved'] + utils.flatten(sample['dense_retrieved_hist'])
    all_retrieved.sort(key=lambda k: ps_ratio*k['s1_para_score'] + 
                                     (1-ps_ratio)*k.get('s1_sent_score_max', 0.0), 
                       reverse=True) # sorting unnecessary but is a rough heuristic to minimise looping time below.. 


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
        if i == uptohop:
            break
        if pred_hist[4] > best_score:
            best_score = pred_hist[4]
            best_hop = i
            sample['best_hop'] = best_hop + 1
            sample['total_hops'] = len(sample['s2_hist_all'])
            sample['s2_best'] = sample['s2_hist_all'][best_hop]
            para_list = []
            para_dict = {}  # {'titleA': [{para 1}, {para 2}, ...]}
            para_dict_compressed = {}
            idx_set = set()
            for s in sample['s2_best']:
                s['s2ev_score'] = best_score  # early bug had s2_hist ev scores all set to the latest one so copy the true s2_evscore back from pred_hist
                
                # add full para text
                if s['idx'] not in idx_set:
                    idx_set.add(s['idx'])
                    foundpara = None
                    for para in all_retrieved:
                        if para['idx'] == s['idx']:
                            foundpara = para
                            break
                    assert foundpara is not None
                    if para_dict.get(foundpara['title']) is None:
                        para_dict[foundpara['title']] = []
                    para_dict[foundpara['title']].append(foundpara)
                    para_list.append(foundpara)
            para_dict[foundpara['title']].sort(key=lambda p: p['idx'])
            sample['para_dict'] = para_dict
            sample['para_list'] = para_list
            for t in para_dict:
                para_dict_compressed[t] = ''
                for para in para_dict[t]:
                    p = para['text'].strip()
                    if p[-1] not in ['.', '?', ':', ';', '!']:
                        p += '.'
                    para_dict_compressed[t] += ' ' + p
            para_dict_compressed[t] = para_dict_compressed[t].strip()
            sample['para_dict_compressed'] = para_dict_compressed
                
            sample['s2_best_preds'] = {'s2_ans_pred': pred_hist[0], 's2_ans_pred_score':pred_hist[1], 
                                       's2_ans_insuff_score': pred_hist[2], 's2_ans_conf_delta': pred_hist[3],
                                       's2ev_score': pred_hist[4], 's2_best_hop': best_hop}
            
    return


def join_annots(samples, uptohop=-1):
    found = 0
    not_found = []
    for s in samples:
        get_best_hop(s, uptohop)
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
    #print(not_found)
    return


def print_titles(samples):
    all_f1 = []
    all_f1_top3 = []
    for s in samples:
        if s.get('annots'):
            print('QUESTION:', {s['question']})
            print('GOLD:', [t['title'] for t in s['annots']])
            print('RETRIEVED:', [t['title'] for t in s['s2_best']])
            f1 = eval_metrics.get_f1([t['title'] for t in s['s2_best']], [t['title'] for t in s['annots']])
            all_f1.append(f1)
            f1_top3 = eval_metrics.get_f1(list(full510sciencv2[1]['para_dict_compressed'].keys())[:3], 
                                          [t['title'] for t in s['annots']])
            all_f1_top3.append(f1_top3)
            print(f"F1: {f1}  Top3 F1: {f1_top3}")
            print()
    print(f"MEAN F1: {sum(all_f1)/len(all_f1)}")
    print(f"MEAN Top 3 F1: {sum(all_f1_top3)/len(all_f1_top3)}")
    return


def print_retrieved(samples):
    for s in samples:
        if s.get('annots'):
            print('QUESTION:', {s['question']})
            print('RETRIEVED NOT GT:', [(t['title'], t['sentence']) for t in s['s2_best'] if t['title'] not in [x['title'] for x in s['annots']] ])
            print('RETRIEVED GT:', [(t['title'], t['sentence']) for t in s['s2_best'] if t['title'] in [x['title'] for x in s['annots']] ])
    return    


def output_csv(samples, outfile):
    """ Output to csv with columns:
    `,claim,anti-claim,Title_1,Title_2,Title_3,Abstract_1,Abstract_2,Abstract_3,A_Title_1,A_Title_2,A_Title_3,A_Abstract_1,A_Abstract_2,A_Abstract_3
    ` = 0-based idx
    
    claim = even nums, corresponding anticlaim = next odd num
    """
    template = {'`':-1,'claim':'','anti-claim':'','Title_1':'','Title_2':'','Title_3':'','Abstract_1':'','Abstract_2':'','Abstract_3':'',
                'A_Title_1':'','A_Title_2':'','A_Title_3':'','A_Abstract_1':'','A_Abstract_2':'','A_Abstract_3':''}
    outlist = []
    outidx = 0
    for i, s in enumerate(samples):
        if i % 2 == 0:
            out = copy.deepcopy(template)
            out['`'] = outidx
            out['claim'] = s['question']
            for j, title in enumerate(s['para_dict_compressed']):
                if j > 2:
                    break
                outtitle = j + 1
                key = 'Title_' + str(outtitle)
                out[key] = title.strip()
                key = 'Abstract_' + str(outtitle)
                out[key] = s['para_dict_compressed'][title].strip()
        else:
            out['anti-claim'] = s['question']
            for j, title in enumerate(s['para_dict_compressed']):
                if j > 2:
                    break
                outtitle = j + 1
                key = 'A_Title_' + str(outtitle)
                out[key] = title.strip()
                key = 'A_Abstract_' + str(outtitle)
                out[key] = s['para_dict_compressed'][title].strip()
            outlist.append(out)                
            outidx += 1
    utils.write_csv_fromjsonl(outfile, outlist, enc='UTF8', headers=True)
    return outlist


annots14file = '/home/thar011/data/SCI/gold_annots_14_claims.csv'
abstractsfull300 = '/large_data/thar011/out/mdr/logs/SCI_ITER_scifullabstracts_claimstest_test4-01-06-2023-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue/samples_with_context.jsonl'
abstractsfull510 = '/large_data/thar011/out/mdr/logs/SCI_ITER_scifullabstracts_claimstest_test5-01-06-2023-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue/samples_with_context.jsonl'
abstractsfull510sciencoderv1 = '/large_data/thar011/out/mdr/logs/SCI_ITER_scifullabstracts_sciencoderv1_claimstest_test7-01-06-2023-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue/samples_with_context.jsonl'
abstractsfull510sciencoderv2 = '/large_data/thar011/out/mdr/logs/SCI_ITER_scifullabstracts_sciencoderv2_claimstest_test8-01-06-2023-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue/samples_with_context.jsonl'
abstractsparas = '/large_data/thar011/out/mdr/logs/SCI_ITER_sciparas_claimstest_test6-01-06-2023-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue/samples_with_context.jsonl'

wikifull = '/large_data/thar011/out/mdr/logs/SCI_ITER_fullwiki_claimstest_test1-12-14-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue/samples_with_context.jsonl'

df = pd.read_csv(annots14file)  # ['Unnamed: 0', 'claim_id', 'claim', 'label', 'title', 'abstract', 'rationales', 'abstract_id']
annots = {}
for r in df.itertuples():
    c = r.claim
    if c[-1] == '.':
        c = c[:-1]
    if annots.get(c) is None:
        annots[c] = []
    annots[c].append( {'label': r.label, 'title': r.title, 'abstract': r.abstract, 'gold_sents': r.rationales, 'abstract_id': r.abstract_id} )    
    


full300 = utils.load_jsonl(abstractsfull300)   # list of {}: dict_keys(['question', 'answer', 'mc_options', 'init_context', 'src', 'type', '_id', 'dense_retrieved', 's1', 's2', 's2_full', 's2_ans_pred', 's2_ans_pred_score', 's2_ans_insuff_score', 's2_ans_conf_delta', 's2ev_score', 'stop_reason', 'dense_retrieved_hist', 's1_hist', 's2_hist', 's2_pred_hist'])
full510 = utils.load_jsonl(abstractsfull510)   # list of {}: dict_keys(['question', 'answer', 'mc_options', 'init_context', 'src', 'type', '_id', 'dense_retrieved', 's1', 's2', 's2_full', 's2_ans_pred', 's2_ans_pred_score', 's2_ans_insuff_score', 's2_ans_conf_delta', 's2ev_score', 'stop_reason', 'dense_retrieved_hist', 's1_hist', 's2_hist', 's2_pred_hist'])
full510sciencv1 = utils.load_jsonl(abstractsfull510sciencoderv1)   # list of {}: dict_keys(['question', 'answer', 'mc_options', 'init_context', 'src', 'type', '_id', 'dense_retrieved', 's1', 's2', 's2_full', 's2_ans_pred', 's2_ans_pred_score', 's2_ans_insuff_score', 's2_ans_conf_delta', 's2ev_score', 'stop_reason', 'dense_retrieved_hist', 's1_hist', 's2_hist', 's2_pred_hist'])
full510sciencv2 = utils.load_jsonl(abstractsfull510sciencoderv2)   # list of {}: dict_keys(['question', 'answer', 'mc_options', 'init_context', 'src', 'type', '_id', 'dense_retrieved', 's1', 's2', 's2_full', 's2_ans_pred', 's2_ans_pred_score', 's2_ans_insuff_score', 's2_ans_conf_delta', 's2ev_score', 'stop_reason', 'dense_retrieved_hist', 's1_hist', 's2_hist', 's2_pred_hist'])
paras = utils.load_jsonl(abstractsparas)
wiki = utils.load_jsonl(wikifull)

join_annots(full300) # dict_keys(['question', 'answer', 'mc_options', 'init_context', 'src', 'type', '_id', 'dense_retrieved', 's1', 's2', 's2_full', 's2_ans_pred', 's2_ans_pred_score', 's2_ans_insuff_score', 's2_ans_conf_delta', 's2ev_score', 'stop_reason', 'dense_retrieved_hist', 's1_hist', 's2_hist', 's2_pred_hist', 's2_hist_all', 's2_pred_hist_all', 'best_hop', 'total_hops', 's2_best', 's2_best_preds', 'annots'])
join_annots(full510) # dict_keys(['question', 'answer', 'mc_options', 'init_context', 'src', 'type', '_id', 'dense_retrieved', 's1', 's2', 's2_full', 's2_ans_pred', 's2_ans_pred_score', 's2_ans_insuff_score', 's2_ans_conf_delta', 's2ev_score', 'stop_reason', 'dense_retrieved_hist', 's1_hist', 's2_hist', 's2_pred_hist', 's2_hist_all', 's2_pred_hist_all', 'best_hop', 'total_hops', 's2_best', 's2_best_preds', 'annots'])
join_annots(full510sciencv1) # dict_keys(['question', 'answer', 'mc_options', 'init_context', 'src', 'type', '_id', 'dense_retrieved', 's1', 's2', 's2_full', 's2_ans_pred', 's2_ans_pred_score', 's2_ans_insuff_score', 's2_ans_conf_delta', 's2ev_score', 'stop_reason', 'dense_retrieved_hist', 's1_hist', 's2_hist', 's2_pred_hist', 's2_hist_all', 's2_pred_hist_all', 'best_hop', 'total_hops', 's2_best', 's2_best_preds', 'annots'])
join_annots(full510sciencv2) # dict_keys(['question', 'answer', 'mc_options', 'init_context', 'src', 'type', '_id', 'dense_retrieved', 's1', 's2', 's2_full', 's2_ans_pred', 's2_ans_pred_score', 's2_ans_insuff_score', 's2_ans_conf_delta', 's2ev_score', 'stop_reason', 'dense_retrieved_hist', 's1_hist', 's2_hist', 's2_pred_hist', 's2_hist_all', 's2_pred_hist_all', 'best_hop', 'total_hops', 's2_best', 's2_best_preds', 'annots'])
join_annots(paras)
join_annots(wiki)

#print_titles(full300)  # MEAN F1: 0.08714285714285716 orig
print_titles(full300)  # MEAN F1: 0.09285714285714287
print_titles(full510)  # MEAN F1: 0.0935714285714286
print_titles(full510sciencv1)  # MEAN F1: 0.11428571428571431
print_titles(full510sciencv2)  # MEAN F1: 0.14142857142857146
#print_titles(paras) # MEAN F1: 0.12714285714285714
print_titles(paras) # MEAN F1: 0.12714285714285717

#join_annots(full510sciencv1, uptohop=1) # dict_keys(['question', 'answer', 'mc_options', 'init_context', 'src', 'type', '_id', 'dense_retrieved', 's1', 's2', 's2_full', 's2_ans_pred', 's2_ans_pred_score', 's2_ans_insuff_score', 's2_ans_conf_delta', 's2ev_score', 'stop_reason', 'dense_retrieved_hist', 's1_hist', 's2_hist', 's2_pred_hist', 's2_hist_all', 's2_pred_hist_all', 'best_hop', 'total_hops', 's2_best', 's2_best_preds', 'annots'])
#print_titles(full510sciencv1)  # MEAN F1: 0.11000000000000001
#join_annots(full510sciencv1, uptohop=2) # dict_keys(['question', 'answer', 'mc_options', 'init_context', 'src', 'type', '_id', 'dense_retrieved', 's1', 's2', 's2_full', 's2_ans_pred', 's2_ans_pred_score', 's2_ans_insuff_score', 's2_ans_conf_delta', 's2ev_score', 'stop_reason', 'dense_retrieved_hist', 's1_hist', 's2_hist', 's2_pred_hist', 's2_hist_all', 's2_pred_hist_all', 'best_hop', 'total_hops', 's2_best', 's2_best_preds', 'annots'])
#print_titles(full510sciencv1)  # MEAN F1: 0.11071428571428574
#join_annots(full510sciencv1, uptohop=3) # dict_keys(['question', 'answer', 'mc_options', 'init_context', 'src', 'type', '_id', 'dense_retrieved', 's1', 's2', 's2_full', 's2_ans_pred', 's2_ans_pred_score', 's2_ans_insuff_score', 's2_ans_conf_delta', 's2ev_score', 'stop_reason', 'dense_retrieved_hist', 's1_hist', 's2_hist', 's2_pred_hist', 's2_hist_all', 's2_pred_hist_all', 'best_hop', 'total_hops', 's2_best', 's2_best_preds', 'annots'])
#print_titles(full510sciencv1)  # MEAN F1: 0.11285714285714286

join_annots(full510sciencv2, uptohop=1) # dict_keys(['question', 'answer', 'mc_options', 'init_context', 'src', 'type', '_id', 'dense_retrieved', 's1', 's2', 's2_full', 's2_ans_pred', 's2_ans_pred_score', 's2_ans_insuff_score', 's2_ans_conf_delta', 's2ev_score', 'stop_reason', 'dense_retrieved_hist', 's1_hist', 's2_hist', 's2_pred_hist', 's2_hist_all', 's2_pred_hist_all', 'best_hop', 'total_hops', 's2_best', 's2_best_preds', 'annots'])
print_titles(full510sciencv2)  # MEAN F1: 0.12999999999999998
join_annots(full510sciencv2, uptohop=2) # dict_keys(['question', 'answer', 'mc_options', 'init_context', 'src', 'type', '_id', 'dense_retrieved', 's1', 's2', 's2_full', 's2_ans_pred', 's2_ans_pred_score', 's2_ans_insuff_score', 's2_ans_conf_delta', 's2ev_score', 'stop_reason', 'dense_retrieved_hist', 's1_hist', 's2_hist', 's2_pred_hist', 's2_hist_all', 's2_pred_hist_all', 'best_hop', 'total_hops', 's2_best', 's2_best_preds', 'annots'])
print_titles(full510sciencv2)  # MEAN F1: 0.1349999999999999
join_annots(full510sciencv2, uptohop=3) # dict_keys(['question', 'answer', 'mc_options', 'init_context', 'src', 'type', '_id', 'dense_retrieved', 's1', 's2', 's2_full', 's2_ans_pred', 's2_ans_pred_score', 's2_ans_insuff_score', 's2_ans_conf_delta', 's2ev_score', 'stop_reason', 'dense_retrieved_hist', 's1_hist', 's2_hist', 's2_pred_hist', 's2_hist_all', 's2_pred_hist_all', 'best_hop', 'total_hops', 's2_best', 's2_best_preds', 'annots'])
print_titles(full510sciencv2)  # MEAN F1: 0.1392857142857143


print_retrieved(paras)

output_csv(full300, os.path.join(outfile_base, 'abstracts_full300.csv'))
output_csv(full510, os.path.join(outfile_base, 'abstracts_full510.csv'))
output_csv(full510sciencv1, os.path.join(outfile_base, 'abstracts_full510_sciencoderv1.csv'))
output_csv(full510sciencv2, os.path.join(outfile_base, 'abstracts_full510_sciencoderv2.csv'))
output_csv(paras, os.path.join(outfile_base, 'abstracts_paras.csv'))
output_csv(wiki, os.path.join(outfile_base, 'wiki_paras.csv'))


