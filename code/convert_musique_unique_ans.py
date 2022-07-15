#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 15:10:14 2021

@author: tim hartill

Convert Musique (ans) into uqa formatted datasets 

Notes:
1.  Converting musique "ans" versions only, not "full" versions (which contain unanswerable questions where one+ of the decomp steps is unanswerable since no supporting para given)
    musique_ans_v0.1_dev.jsonl is constructed to not have answer overlap with musique_ans_v0.1_train.jsonl
    Therefore we treat musique_ans_v0.1_dev.jsonl as an OOD eval only dataset ("musique_mu_dev_.." 2417 samples) 
    but we create a training dataset from the decomps + paras
    and construct a separate in-domain dev split from musique_ans_v0.1_train.jsonl (382 samples)
    NB: Confusingly we use the term "full" to mean all the musique "ans" train split samples 
        ie (19938-382=19556 train samples) and output these into uqa directories starting with 'musique_full_' 
        vs just those with unique answers (2057 train samples, 382 dev samples) which are output into dirs starting with 'musique_'
2. label format with decomp steps: "<s> the answer ## decomp step 1? decomp ans ## decomp step 2? decomp ans ## decomp step 3? decomp ans </s>"
3. This version, in contrast to (deprecated) convert_musique.py creates a train set of musique samples with unique answers (plus "full" versions)
   and a new dev set which has unique answers within dev and with an answer+last decomp that's not in train. (2057 train samples, 382 dev samples)
   This is because Musique contains a lot of very similar questions with the same answer and the same final decomp step
   which might make it possible for a model to learn a probabalistic bias.
4. Added output of "explanations" versions which are formed by removing the '>>', format_sentencing both sub q and sub ans and randomly shuffling the (pairs) of sentences.
    od_ans: q \\n \t ans
    od_expl: Add explanation: q \\n \t explanation
    expl_ans: q \\n explanation \t ans
5. Added output of self-supervised facts dataset in "dev in train" form..


Initially developed against Musique v0.1. 
Subsequently v1.0 was released which includes the test set but train/dev are identical to v0.1.

- Multiple paras in 'paragraphs' have same title & probably from same doc eg "Green" in mu_dev[0].
- 'answer_aliases' key contains [alternative answers] whereas 'answer' contains a string. 

"""

import os
import copy
import random
import numpy as np
from collections import Counter
from utils import load_jsonl, create_uqa_example, format_decomp_ans, load_model, string_to_ids
from text_processing import white_space_fix, replace_chars, format_sentence

Q_PREFIX = 'Add Explanation: '
selfsupervisedkey = '_selfsvised'

UQA_DIR = '/data/thar011/data/unifiedqa/'

MU_DIR_IN = '/home/thar011/data/musique/musique_v1.0/data/'  # v0.1:'/home/thar011/data/musique/musique_v0.1/'
MU_TRAIN_FILE = 'musique_ans_v1.0_train.jsonl'
MU_DEV_FILE = 'musique_ans_v1.0_dev.jsonl'


#MUFULL_TRAIN_FILE = 'musique_full_v0.1_train.jsonl'
#mufull_train = load_jsonl(os.path.join(MU_DIR_IN, MUFULL_TRAIN_FILE))

mu_train = load_jsonl(os.path.join(MU_DIR_IN, MU_TRAIN_FILE))  #dict_keys(['id', 'paragraphs', 'question', 'question_decomposition', 'answer', 'answer_aliases', 'answerable'])
mu_dev = load_jsonl(os.path.join(MU_DIR_IN, MU_DEV_FILE))

def calc_unique_answers(mu_data):
    """ See how many unique answers there are
    + unique paras ...
    """
    all_answers = []
    all_answers_last_decomp = []
    all_titles = set()
    all_title_idxs = set()
    for mu_sample in mu_data:
        all_answers.append(mu_sample['answer'].lower().strip())
        all_answers_last_decomp.append(mu_sample['answer'].lower().strip() + mu_sample['question_decomposition'][-1]['question'].lower().strip())
        for para in mu_sample['paragraphs']:
            if para['is_supporting']:
                all_titles.add(para['title'].lower())
                all_title_idxs.add( (para['title'].lower(), para['paragraph_text'].lower()) )
        
    unique_answers = set(all_answers)
    unique_ans_last_decomp = set(all_answers_last_decomp)
    print(f"Total count: {len(all_answers)}  Unique answers: {len(unique_answers)}  Unique ans+Last decomp: {len(unique_ans_last_decomp)}")
    print(f"#Unique titles:{len(all_titles)}  #Unique paras:{len(all_title_idxs)}")
    ans_counts = Counter(all_answers)
    print(f'Top 20 Most common answers: {ans_counts.most_common(20)}')
    print(f'20 Least common answers: {ans_counts.most_common()[::-1][:20]}')    
    return ans_counts

ans_counts_train = calc_unique_answers(mu_train)  # Total count: 19938  Unique answers: 2057  Unique ans+Last decomp: 3056
ans_counts_dev = calc_unique_answers(mu_dev)  # Total count: 2417  Unique answers: 738  Unique ans+Last decomp: 841

#mu_sample = mu_train[0]
#mu_sample.keys()  # dict_keys(['id', 'paragraphs', 'question', 'question_decomposition', 'answer', 'answer_aliases', 'answerable'])
#mu_sample['id'] # '2hop__482757_12019'
#mu_sample['paragraphs'][0].keys() # dict_keys(['idx', 'title', 'paragraph_text', 'is_supporting'])
#mu_sample['paragraphs'][0]['idx'] # 0
#mu_sample['paragraphs'][0]['paragraph_text']
#mu_sample['question'] # overall question
#mu_sample['answer'] # overall answer
#mu_sample['answer_aliases']

#mu_sample['question_decomposition']
#mu_sample['question_decomposition'][0].keys() # list of decomps each of dict_keys(['id', 'question', 'answer', 'paragraph_support_idx'])
#mu_sample['question_decomposition'][0]['question']
#mu_sample['question_decomposition'][0]['answer']  # last answer same as mu_sample['answer']
#mu_sample['question_decomposition'][0]['paragraph_support_idx']
#mu_sample['paragraphs'][ mu_sample['question_decomposition'][0]['paragraph_support_idx'] ]['paragraph_text']
#mu_sample['paragraphs'][ mu_sample['question_decomposition'][0]['paragraph_support_idx'] ]['title']


#tokenizer = load_model(loadwhat='tokenizer_only')
#tok_counts = []
#for i, mu_sample in enumerate(mu_train):
#    question = mu_sample['question']
#    paras = retrieve_paras(mu_sample)
#    context = white_space_fix(' '.join(paras))
#    input_string = create_uqa_example(question, context)    
#    ids = string_to_ids(input_string, tokenizer, verbose=False)
#    tok_counts.append( len(ids) )
#    if i % 1000 == 0:
#        print(f'Processed: {i}')
#tok_counts_np = np.array(tok_counts)
#print(f"count:{len(tok_counts)} max:{tok_counts_np.max()} mean:{tok_counts_np.mean()}") # count:19938 max:512 mean:299.1696759955863
#hittoklimit = np.where(tok_counts_np >= 512)
#hittoklimit[0].shape  # (1928,)  Approx 10% will be truncated..

#d_max = -1
#for i, mu_sample in enumerate(mu_train):
#    d_num = len(mu_sample['question_decomposition'])
#    if mu_sample['question_decomposition'][0]['question'].find('#9') != -1:
#        print(mu_sample)
#    if d_num > d_max:
#        d_max = d_num
#print(f"Max decompositions: {d_max}")


def retrieve_paras(mu_sample):
    """ Return list of paragraphs supporting each decomp step (1 para per decomp step so indices match)
    """
    paras = []
    for decomp in mu_sample['question_decomposition']:
        para_idx = decomp['paragraph_support_idx']
        if para_idx is not None:
            para = mu_sample['paragraphs'][para_idx]['paragraph_text'].strip()
            if para[-1] not in ['.', '?', '!']:
                para += '.'
            paras.append(replace_chars(para))
        else:
            paras.append('')  # for musique full, keep paras/decomps idxs aligned
    return paras


def process_musique(mu_data, make_all_dev=True):
    """ Retrieve, format and create new keys for outputting all the datasets, namely:
    from input: dict_keys(['id', 'paragraphs', 'question', 'question_decomposition', 'answer', 'answer_aliases', 'answerable'])
    adds keys to output:
        dict_keys(['id', 'paragraphs', 'question', 'question_decomposition', 'answer', 'answer_aliases', 'answerable', 
                   'split', 'context_paras', 'decomp_ans_str', 'decomp_context', 'explanation', 'ans_status'])
    'split',            : 'train' = unique ans (& full) train split  'unassigned'= full train split only  'dev'=new dev split (or orig dev musique_mu_)
    'context_paras',    : string concatenation of pos paras with minor preprocessing
    'decomp_ans_str',   : string of form "overall answer ## subq1? ans1. ## subq2? ans2. ## ..."
    'decomp_context',   : string of form "sub1. ans1. subq2. ans2. subq3?" ie no final answer
    'explanation',      : str of form "subq1. ans1. subq2. ans2. ..."
    'ans_status'        : 'first'=1st row with unique answer or 'not_first'
    
    
    'question_decomposition' key input: list of dict_keys(['id', 'question', 'answer', 'paragraph_support_idx'])
    adds keys to get: dict_keys(['id', 'question', 'answer', 'paragraph_support_idx', 'question_subst', 'context_para', 'title', 'text'])
    eg: 
        {'id': 8966, 'question': '#1 was a president of what country?', 'answer': 'U.S.', 'paragraph_support_idx': 5,
         'question_subst': 'Nixon was a president of what country?',
         'context_para': "(Josip Broz Tito) Yugoslavia had a liberal travel policy permitting foreigners to freely travel through the country and its citizens to travel worldwide, whereas it was limited by most Communist countries. A number[quantify] of Yugoslav citizens worked throughout Western Europe. Tito met many world leaders during his rule, such as Soviet rulers Joseph Stalin, Nikita Khrushchev and Leonid Brezhnev; Egypt's Gamal Abdel Nasser, Indian politicians Jawaharlal Nehru and Indira Gandhi; British Prime Ministers Winston Churchill, James Callaghan and Margaret Thatcher; U.S. Presidents Dwight D. Eisenhower, John F. Kennedy, Richard Nixon, Gerald Ford and Jimmy Carter; other political leaders, dignitaries and heads of state that Tito met at least once in his lifetime included Che Guevara, Fidel Castro, Yasser Arafat, Willy Brandt, Helmut Schmidt, Georges Pompidou, Queen Elizabeth II, Hua Guofeng, Kim Il Sung, Sukarno, Sheikh Mujibur Rahman, Suharto, Idi Amin, Haile Selassie, Kenneth Kaunda, Gaddafi, Erich Honecker, Nicolae Ceaușescu, János Kádár and Urho Kekkonen. He also met numerous celebrities.",
         'title': Josip Broz Tito,
         'text': "Yugoslavia had a liberal travel policy permitting foreigners to freely travel through the country and its citizens to travel worldwide, whereas it was limited by most Communist countries. A number[quantify] of Yugoslav citizens worked throughout Western Europe. Tito met many world leaders during his rule, such as Soviet rulers Joseph Stalin, Nikita Khrushchev and Leonid Brezhnev; Egypt's Gamal Abdel Nasser, Indian politicians Jawaharlal Nehru and Indira Gandhi; British Prime Ministers Winston Churchill, James Callaghan and Margaret Thatcher; U.S. Presidents Dwight D. Eisenhower, John F. Kennedy, Richard Nixon, Gerald Ford and Jimmy Carter; other political leaders, dignitaries and heads of state that Tito met at least once in his lifetime included Che Guevara, Fidel Castro, Yasser Arafat, Willy Brandt, Helmut Schmidt, Georges Pompidou, Queen Elizabeth II, Hua Guofeng, Kim Il Sung, Sukarno, Sheikh Mujibur Rahman, Suharto, Idi Amin, Haile Selassie, Kenneth Kaunda, Gaddafi, Erich Honecker, Nicolae Ceaușescu, János Kádár and Urho Kekkonen. He also met numerous celebrities."}        
    """
    unique_ans = {}
    decomp_len_counts_train = {}
    tr_cnt = 0
    unique_ans_last_decomp = set()
    for i, mu_sample in enumerate(mu_data):
        if make_all_dev:
            mu_sample['split'] = 'dev'
        else:    
            mu_sample['split'] = 'unassigned'
        question = mu_sample['question'].strip()
        answer = mu_sample['answer'].strip()
        paras = retrieve_paras(mu_sample)
        mu_sample['context_paras'] = white_space_fix(' '.join(paras))
        decomp_ans_str = ''
        prior_answers = []
        for j, decomp_step in enumerate(mu_sample['question_decomposition']):   # list of decomps each of dict_keys(['id', 'question', 'answer', 'paragraph_support_idx'])
            this_ans_str, prior_answers, subst_decomp = format_decomp_ans(decomp_step['question'], decomp_step['answer'], j, prior_answers)
            decomp_ans_str += this_ans_str
            decomp_step['question_subst'] = subst_decomp
            para_idx = decomp_step['paragraph_support_idx']
            if para_idx is not None:
                title = replace_chars(mu_sample['paragraphs'][para_idx]['title'].strip())
                text = replace_chars(white_space_fix(mu_sample['paragraphs'][para_idx]['paragraph_text'].strip()))
                if text[-1] not in ['.', '?', '!']:
                    text += '.'
                decomp_para = '(' + title + ') ' + text
            else:
                title = ''
                text = ''
                decomp_para = ''
            decomp_step['context_para'] = decomp_para
            decomp_step['title'] = title
            decomp_step['text'] = text
        
        mu_sample['decomp_ans_str'] = answer + decomp_ans_str
        dc_len = len(decomp_ans_str)
        last_ans_start = dc_len - decomp_ans_str[::-1].find('?')
        mu_sample['decomp_context'] = decomp_ans_str[:last_ans_start].replace('##','.').replace(' .','.')[2:]

        # reformat decomp into explanation:
        facts = decomp_ans_str.split(' ## ')
        facts = [f for f in facts if f.strip() != '']
        for i, f in enumerate(facts):
            qa = f.split('? ')
            if len(qa) == 2 : # otherwise multiple '?', use as-is without processing
                q = format_sentence(qa[0], endchar='?', strip=['>>'])
                a = format_sentence(qa[1], endchar='.')
                facts[i] = q + ' ' + a
        random.shuffle(facts)
        mu_sample['explanation'] = ' '.join(facts)

        ans = mu_sample['answer'].lower().strip()
        ans_last_decomp = ans + mu_sample['question_decomposition'][-1]['question_subst'].lower().strip()

        if unique_ans.get(ans) is None:
            mu_sample['ans_status'] = 'first'
            unique_ans[ans] = 1
            unique_ans_last_decomp.add(ans_last_decomp) # there are many repeated samples with same ans + same last decomp
            if not make_all_dev:
                mu_sample['split'] = 'train'             
        else:
            mu_sample['ans_status'] = 'not_first'
            unique_ans[ans] += 1

        if mu_sample['split'] in ['train', 'dev']:
            tr_cnt += 1
            l = len(mu_sample['question_decomposition'])
            if decomp_len_counts_train.get(l) is None:
                decomp_len_counts_train[l] = 1
            else:
                decomp_len_counts_train[l] += 1
        
        if i % 1000 == 0:
            print(f'Processed: {i}')
    if not make_all_dev:
        desc = 'Train'
    else:
        desc = 'Dev'
    print(f"Finished Assigning {desc}. Count: {tr_cnt}")
    print(f"{desc} Decomp length distribution: {decomp_len_counts_train}")

    if not make_all_dev:
        dev_answers = set()
        dev_cnt = 0
        decomp_len_counts_dev = {}
        
        print("Making dev set with unassigned samples with unique answers over dev")
        for i, mu_sample in enumerate(mu_data[::-1]):
            if mu_sample['split'] == 'unassigned':
                ans = mu_sample['answer'].lower().strip()
                ans_last_decomp = ans + mu_sample['question_decomposition'][-1]['question_subst'].lower().strip()
                if ans not in dev_answers and ans_last_decomp not in unique_ans_last_decomp:
                    dev_answers.add(ans)
                    mu_sample['split'] = 'dev'
                    dev_cnt += 1
                    l = len(mu_sample['question_decomposition'])
                    if decomp_len_counts_dev.get(l) is None:
                        decomp_len_counts_dev[l] = 1
                    else:
                        decomp_len_counts_dev[l] += 1    
            if i % 1000 == 0:
                print(f'Processed: {i} Current dev count: {dev_cnt}')
        print(f"Finished Assigning New Dev. Dev count: {dev_cnt}")
        print(f"New Dev Decomp length distribution: {decomp_len_counts_dev}")  
        decomp_len_counts_train_full = {}
        tr_cnt_full = 0
        for i, mu_sample in enumerate(mu_data):
            if mu_sample['split'] != 'dev':
                tr_cnt_full += 1
                l = len(mu_sample['question_decomposition'])
                if decomp_len_counts_train_full.get(l) is None:
                    decomp_len_counts_train_full[l] = 1
                else:
                    decomp_len_counts_train_full[l] += 1    
        print(f"FULL train count: {tr_cnt_full}")
        print(f"FULL train Decomp length distribution: {decomp_len_counts_train_full}")
    return


def get_paras(mu_data):
    """ Return unique paragraphs and paragraph titles as sets
    """
    paras = []
    titles = []
    for i, mu_sample in enumerate(mu_data):
        if mu_sample['split'] in ['dev','train','unassigned']:
            for j, decomp_step in enumerate(mu_sample['question_decomposition']):   # list of decomps each of dict_keys(['id', 'question', 'answer', 'paragraph_support_idx'])
                paras.append( decomp_step['context_para'] )
                titles.append( mu_sample['paragraphs'][decomp_step['paragraph_support_idx']]['title'] )
    print(f" Number of paras/titles: {len(titles)}") 
    paras = set(paras)
    titles = set(titles)
    print(f"Number unique paras: {len(paras)}  Number unique titles: {len(titles)}")
    return paras, titles


def get_facts_datasets(mu_data, train_splits = ['train'], dataset_format = 'qa'):
    """ Output facts datasets:
    dataset_format = 'qa':
        musique_decomp_train: decomp q \\n para \t decomp ans                   train=decomps from mu train; dev=decomps from new dev (setting where qa dataset train can see it's facts but dev can't)
        musique_decomp_new_dev_in_train: decomp q \\n para \t decomp ans        train=decomps from mu train + decomps from new dev; dev=decomps from new dev (setting where qa dataset is allowed to have seen train & dev facts)
        musique_mu_dev_decomp: decomp q \\n para \t decomp ans                  train=decomps from musique dev; dev=same (dev is just to check how well decomps are learned) 
        musique_decomp_all_dev_in_train decomp q \\n para \t decomp ans         train=decomps from mu train + decomps from new dev + decomps from musique dev; dev=decomps from musique dev
    dataset_format = 'anything else':
        all: para \\n \n
    """
    train_list = []
    dev_list = []
    for i, mu_sample in enumerate(mu_data):
        for j, decomp_step in enumerate(mu_sample['question_decomposition']):   # list of decomps each of dict_keys(['id', 'question', 'answer', 'paragraph_support_idx', 'question_subst', 'context_para'])
            if dataset_format == 'qa':
                sample = create_uqa_example(decomp_step['question_subst'], decomp_step['context_para'], decomp_step['answer'].strip() )
            else:
                sample = create_uqa_example(decomp_step['context_para'], ' ', None, append_q_char='.')
            if mu_sample['split'] == 'dev':
                dev_list.append(sample)
            elif mu_sample['split'] in train_splits:
                train_list.append(sample)
        if i % 1000 == 0:
            print(f'Processed: {i}')      
    train_list = list(set(train_list))
    dev_list = list(set(dev_list))
    print(f"Unique Train Decomps:{len(train_list)}  Unique Dev Decomps:{len(dev_list)}")
    return train_list, dev_list


def get_qa_datasets(mu_data, train_splits = ['train']):
    """ Output qa datasets:
    NB: if train_splits = ['train','unassigned']  will output "full" datasets vs 'unique_ans_ versions' if train_splits = ['train'] 
        
    musique_qa: q \\n \t ans                                                train=mu train, dev=new dev from mu train
    musique_qa_paras: q \\n paras \t ans                                    train=mu train, dev=new dev from mu train
    musique_qa_decomp_ans: Add decomp: q \\n \t ans ## decomps+ans ## ..                train=mu train, dev=new dev from mu train
    musique_qa_paras_decomp_ans: Add decomp: q \\n paras \t ans  ## decomps+ans ## ..   train=mu train, dev=new dev from mu train
    
    musique_mu_dev_qa: q \\n \t ans                                                dev only = musique dev
    musique_mu_dev_qa_paras: q \\n paras \t ans                                    dev only = musique dev
    musique_mu_dev_qa_decomp_ans: Add decomp: q \\n \t ans ## decomps+ans ## ..                dev only = musique dev
    musique_mu_dev_qa_paras_decomp_ans: Add decomp: q \\n paras \t ans  ## decomps+ans ## ..   dev only = musique dev
    
    musique_qa_full_od_ans: q \\n \t ans
    musique_qa_full_od_expl: Add explanation: q \\n \t explanation
    musique_qa_full_expl_ans: q \\n explanation \t ans

    musique_mu_dev_qa_od_ans: q \\n \t ans
    musique_mu_dev_qa_od_expl: Add explanation: q \\n \t explanation
    musique_mu_dev_qa_expl_ans: q \\n explanation \t ans

    """
    ds_template = {'train':[], 'dev':[]}
    mu_qa_dict = {'qa':copy.deepcopy(ds_template),
                  'qa_paras':copy.deepcopy(ds_template),
                  'qa_decomp_ans':copy.deepcopy(ds_template),
                  'qa_paras_decomp_ans':copy.deepcopy(ds_template), 
                  'qa_decomp_context':copy.deepcopy(ds_template),
                  'qa_od_ans':copy.deepcopy(ds_template),
                  'qa_od_expl':copy.deepcopy(ds_template),
                  'qa_expl_ans':copy.deepcopy(ds_template),
                 }

    for i, mu_sample in enumerate(mu_data):
        key =  mu_sample['split']
        if key in train_splits:
            key = 'train'
        if key in ['dev','train']:
            sample = create_uqa_example(mu_sample['question'], None, mu_sample['answer'] )    
            mu_qa_dict['qa'][key].append(sample)
            mu_qa_dict['qa_od_ans'][key].append(sample) #keep separate for consistency..
            sample = create_uqa_example(mu_sample['question'], mu_sample['context_paras'], mu_sample['answer'] )    
            mu_qa_dict['qa_paras'][key].append(sample)
            sample = create_uqa_example("add decomp: " + mu_sample['question'], None, mu_sample['decomp_ans_str'] )    
            mu_qa_dict['qa_decomp_ans'][key].append(sample)
            sample = create_uqa_example("add decomp: " + mu_sample['question'], mu_sample['context_paras'], mu_sample['decomp_ans_str'] )
            mu_qa_dict['qa_paras_decomp_ans'][key].append(sample) 
            sample = create_uqa_example(mu_sample['question'], mu_sample['decomp_context'], mu_sample['answer'] )
            mu_qa_dict['qa_decomp_context'][key].append(sample)
            sample = create_uqa_example(Q_PREFIX + mu_sample['question'], None, mu_sample['explanation'] )
            mu_qa_dict['qa_od_expl'][key].append(sample)
            sample = create_uqa_example(mu_sample['question'], mu_sample['explanation'], mu_sample['answer'] )
            mu_qa_dict['qa_expl_ans'][key].append(sample)
        if i % 1000 == 0:
            print(f'Processed: {i}')
    print(f"Train count: {len(mu_qa_dict['qa']['train'])}  Dev count:{len(mu_qa_dict['qa']['dev'])}")
    return mu_qa_dict


def get_retriever_datasets():
    """ Output training datasets of the form: 
        [ dict_keys(['question', 'answers', 'src', 'type', '_id', 'bridge', 'num_hops', 'pos_paras', 'neg_paras']) ]
        
        type = 'multi'
        bridge is of form [[paratitle0], [paratitle1], ..] ie must find para0 from q before finding para1 from q+para0 etc
    """
    #non-unique (neg or pos) para titles? 560 pos samples like this! MAKE unique with title(2) - pos only. negs can duplicate ok

    # any way to order paras other than sequentially?
    #  YES: 779 samples in train where there is no #1 in question_decomp[1]. checked and both 1&2 reachable from question
    #       400 samples in train where no #x in question_decomp[2] (but there is in 1) - 
    # RULE: any decomp w/o a #x is reachable from the question so can be in bridge[0] list
    #       then add others sequentially ie forall decomps:  if #x in it append to bridge, else append to bridge[0]
    # There are NO 3 hop samples where the last hop doesnt have a #2 and no 4 hop without last hop having a #3 in it
    # SO only need to code RULE

    #TODO add 'answer' -> 'answers' and output that
    #TODO get more negative paras? Just use theirs?...
    
    
    
    

# Create new dev set & preprocess data into convenient format..
# order by number of decomp steps so when creating train set with unique answers we tend to pick the longest instead of discarding 
mu_train = sorted(mu_train, key=lambda mu_sample: len(mu_sample['question_decomposition']), reverse=True)

#num_q = len(mu_train)
#dev_size = int(num_q*0.1)
#np.random.seed(42)
#dev_indices = np.random.choice(num_q, dev_size, replace=False)
process_musique(mu_train, make_all_dev=False)
process_musique(mu_dev, make_all_dev=True)

#check number of mu dev paras are in mu train
train_paras, train_titles = get_paras(mu_train)
# over train/dev/unassigned:
# Number of paras/titles: 46613
#Number unique paras: 13672  Number unique titles: 12304    
# over train/dev only:
#Number of paras/titles: 6158
#Number unique paras: 3957  Number unique titles: 3271  titles are duplicated over difft paras..
mudev_paras, mudev_titles = get_paras(mu_dev)
#Number of paras/titles: 6404
#Number unique paras: 2629  Number unique titles: 2328

common_paras = train_paras.intersection(mudev_paras)
print(f"Number of mu dev paras in train: {len(common_paras)}")  # 0 whether or not include unassigned yay!




### BELOW IS UNMODIFIED FROM 2021 experiments  ######

# Create and output mu_train decomp fact datasets
train_list, dev_list = get_facts_datasets(mu_train)
both_list = train_list + dev_list
print(f'Train: {len(train_list)}  Dev: {len(dev_list)}  Dev in train:{len(both_list)}')  # Train: 3637  Dev: 759  Dev in train:4396

outdir = os.path.join(UQA_DIR, 'musique_decomp_train')
print(f"Creating {outdir}")
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, 'train.tsv')
with open(outfile, 'w') as f:
    f.write(''.join(train_list))
outfile = os.path.join(outdir, 'dev.tsv')
with open(outfile, 'w') as f:
    f.write(''.join(dev_list))

outdir = os.path.join(UQA_DIR, 'musique_decomp_new_dev_in_train')
print(f"Creating {outdir}")
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, 'train.tsv')
with open(outfile, 'w') as f:
    f.write(''.join(both_list)) # mu train + new dev but not mu dev..
outfile = os.path.join(outdir, 'dev.tsv')
with open(outfile, 'w') as f:
    f.write(''.join(dev_list))

# create FULL train decomp fact datasets
full_train_list, full_dev_list = get_facts_datasets(mu_train, train_splits=['train','unassigned'])
full_both_list = full_train_list + full_dev_list
print(f'FULL Train: {len(full_train_list)}  Dev: {len(full_dev_list)}  Dev in train:{len(full_both_list)}')  # FULL Train: 14646  Dev: 759  Dev in train:15405

outdir = os.path.join(UQA_DIR, 'musique_decomp_new_dev_in_train_full')
print(f"Creating {outdir}")
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, 'train.tsv')
with open(outfile, 'w') as f:
    f.write(''.join(full_both_list)) # mu train + new dev but not mu dev..
outfile = os.path.join(outdir, 'dev.tsv')
with open(outfile, 'w') as f:
    f.write(''.join(full_dev_list)) # note samples in difft order due to set() but same actual samples as musique_decomp_new_dev_in_train dev.tsv


# Create and output mu_dev decomp fact datasets
train_list, dev_list = get_facts_datasets(mu_dev)
print(f'Train: {len(train_list)}  Dev: {len(dev_list)}')  # Train: 0  Dev: 2821

outdir = os.path.join(UQA_DIR, 'musique_mu_dev_decomp')
print(f"Creating {outdir}")
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, 'train.tsv')
with open(outfile, 'w') as f:
    f.write(''.join(dev_list))
outfile = os.path.join(outdir, 'dev.tsv')
with open(outfile, 'w') as f:
    f.write(''.join(dev_list)) # dev same as train..

both_list = both_list + dev_list  #All facts both mu train + new dev + mu dev
print(len(both_list))  # 7217
outdir = os.path.join(UQA_DIR, 'musique_decomp_all_dev_in_train')
print(f"Creating {outdir}")
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, 'train.tsv')
with open(outfile, 'w') as f:
    f.write(''.join(both_list))  #All facts both mu train + new dev + mu dev
outfile = os.path.join(outdir, 'dev.tsv')
with open(outfile, 'w') as f:
    f.write(''.join(dev_list))


# Create/output mu_train_full + mu_dev fact datasets in self supervised format
train_list, dev_list = get_facts_datasets(mu_train, train_splits=['train','unassigned'], dataset_format='ssvise')
both_list = list(set(train_list + dev_list))  #13672

outdir = os.path.join(UQA_DIR, 'musique_full_new_dev_in_train'+selfsupervisedkey)
print(f"Creating {outdir}")
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, 'train.tsv')
with open(outfile, 'w') as f:
    f.write(''.join(both_list)) # mu train + new dev but not mu dev..
outfile = os.path.join(outdir, 'dev.tsv')
with open(outfile, 'w') as f:
    f.write(''.join(dev_list)) # note samples in difft order due to set() but same actual samples as musique_decomp_new_dev_in_train dev.tsv


train_list_mudev, dev_list_mudev = get_facts_datasets(mu_dev, train_splits=['train','unassigned'], dataset_format='ssvise')
#train_list_mudev=[], len(dev_list_mudev) = 2629

outdir = os.path.join(UQA_DIR, 'musique_mu_dev'+selfsupervisedkey)
print(f"Creating {outdir}")
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, 'train.tsv')
with open(outfile, 'w') as f:
    f.write(''.join(dev_list_mudev)) # mu dev only..
outfile = os.path.join(outdir, 'dev.tsv')
with open(outfile, 'w') as f:
    f.write(''.join(dev_list_mudev)) # also mu dev only..

all_list = list(set(both_list + dev_list_mudev))  #16301
outdir = os.path.join(UQA_DIR, 'musique_full_all_dev_in_train'+selfsupervisedkey)
print(f"Creating {outdir}")
os.makedirs(outdir, exist_ok=True)
outfile = os.path.join(outdir, 'train.tsv')
with open(outfile, 'w') as f:
    f.write(''.join(all_list)) # mu train + new dev + mu dev..
outfile = os.path.join(outdir, 'dev.tsv')
with open(outfile, 'w') as f:
    f.write(''.join(dev_list)) # 'new' dev


# create and ouput mu_train qa datasets
mu_qa_dict = get_qa_datasets(mu_train)  #Train count: 2057  Dev count:382

for k in mu_qa_dict.keys():   #dict_keys(['qa', 'qa_paras', 'qa_decomp_ans', 'qa_paras_decomp_ans', 'qa_decomp_context', 'qa_od_ans', 'qa_od_expl', 'qa_expl_ans'])
    ds = 'musique_' + k
    outdir = os.path.join(UQA_DIR, ds)
    print(f"Creating {outdir}")
    os.makedirs(outdir, exist_ok=True)    
    outfile = os.path.join(outdir, 'train.tsv')
    with open(outfile, 'w') as f:
        f.write(''.join(mu_qa_dict[k]['train']))
    outfile = os.path.join(outdir, 'dev.tsv')
    with open(outfile, 'w') as f:
        f.write(''.join(mu_qa_dict[k]['dev']))
mu_qa_dict['qa_plus_qa_decomp_ans'] = {'train':copy.deepcopy(mu_qa_dict['qa']['train']) + copy.deepcopy(mu_qa_dict['qa_decomp_ans']['train']), 
                                       'dev':copy.deepcopy(mu_qa_dict['qa']['dev']) + copy.deepcopy(mu_qa_dict['qa_decomp_ans']['dev'])}
mu_qa_dict['qa_paras_plus_qa_paras_decomp_ans'] = {'train':copy.deepcopy(mu_qa_dict['qa_paras']['train']) + copy.deepcopy(mu_qa_dict['qa_paras_decomp_ans']['train']), 
                                                   'dev':copy.deepcopy(mu_qa_dict['qa_paras']['dev']) + copy.deepcopy(mu_qa_dict['qa_paras_decomp_ans']['dev'])}
for k in ['qa_plus_qa_decomp_ans', 'qa_paras_plus_qa_paras_decomp_ans']:
    ds = 'musique_' + k
    outdir = os.path.join(UQA_DIR, ds)
    print(f"Creating {outdir}")
    os.makedirs(outdir, exist_ok=True)    
    outfile = os.path.join(outdir, 'train.tsv')
    with open(outfile, 'w') as f:
        f.write(''.join(mu_qa_dict[k]['train']))
    outfile = os.path.join(outdir, 'dev.tsv')
    with open(outfile, 'w') as f:
        f.write(''.join(mu_qa_dict[k]['dev']))

# create and ouput mu_train FULL qa datasets
# dict_keys(['qa', 'qa_paras', 'qa_decomp_ans', 'qa_paras_decomp_ans', 'qa_decomp_context', 'qa_od_ans', 'qa_od_expl', 'qa_expl_ans'])
mu_qa_dict = get_qa_datasets(mu_train, train_splits=['train','unassigned'])  # Train count: 19556  Dev count:382
for k in ['qa', 'qa_paras']:
    ds = 'musique_' + k + '_full'
    outdir = os.path.join(UQA_DIR, ds)
    print(f"Creating {outdir}")
    os.makedirs(outdir, exist_ok=True)    
    outfile = os.path.join(outdir, 'train.tsv')
    with open(outfile, 'w') as f:
        f.write(''.join(mu_qa_dict[k]['train']))
    outfile = os.path.join(outdir, 'dev.tsv')
    with open(outfile, 'w') as f:
        f.write(''.join(mu_qa_dict[k]['dev']))
    
    
mu_qa_dict['qa_plus_qa_decomp_ans_full'] = {'train':copy.deepcopy(mu_qa_dict['qa']['train']) + copy.deepcopy(mu_qa_dict['qa_decomp_ans']['train']), 
                                       'dev':copy.deepcopy(mu_qa_dict['qa']['dev']) + copy.deepcopy(mu_qa_dict['qa_decomp_ans']['dev'])}
mu_qa_dict['qa_paras_plus_qa_paras_decomp_ans_full'] = {'train':copy.deepcopy(mu_qa_dict['qa_paras']['train']) + copy.deepcopy(mu_qa_dict['qa_paras_decomp_ans']['train']), 
                                                   'dev':copy.deepcopy(mu_qa_dict['qa_paras']['dev']) + copy.deepcopy(mu_qa_dict['qa_paras_decomp_ans']['dev'])}
for k in ['qa_plus_qa_decomp_ans_full', 'qa_paras_plus_qa_paras_decomp_ans_full']:
    ds = 'musique_' + k
    outdir = os.path.join(UQA_DIR, ds)
    print(f"Creating {outdir}")
    os.makedirs(outdir, exist_ok=True)    
    outfile = os.path.join(outdir, 'train.tsv')
    with open(outfile, 'w') as f:
        f.write(''.join(mu_qa_dict[k]['train']))
    outfile = os.path.join(outdir, 'dev.tsv')
    with open(outfile, 'w') as f:
        f.write(''.join(mu_qa_dict[k]['dev']))

#output mu_train full qa explanation datasets:
for k in ['qa_od_ans', 'qa_od_expl', 'qa_expl_ans']:
    ds = 'musique_full_' + k  # note change in naming convention
    outdir = os.path.join(UQA_DIR, ds)
    print(f"Creating {outdir}")
    os.makedirs(outdir, exist_ok=True)    
    outfile = os.path.join(outdir, 'train.tsv')
    with open(outfile, 'w') as f:
        f.write(''.join(mu_qa_dict[k]['train']))
    outfile = os.path.join(outdir, 'dev.tsv')
    with open(outfile, 'w') as f:
        f.write(''.join(mu_qa_dict[k]['dev']))



print('Finished outputting mu_train qa datasets...')

# create and ouput mu_dev qa datasets
mu_qa_dict = get_qa_datasets(mu_dev)  #Train count: 0  Dev count:2417
for k in mu_qa_dict.keys():   #['qa', 'qa_paras', 'qa_decomp_ans', 'qa_paras_decomp_ans', 'qa_decomp_context', 'qa_od_ans', 'qa_od_expl', 'qa_expl_ans']
    ds = 'musique_mu_dev_' + k
    outdir = os.path.join(UQA_DIR, ds)
    print(f"Creating {outdir}")
    os.makedirs(outdir, exist_ok=True)    
    outfile = os.path.join(outdir, 'dev.tsv')
    with open(outfile, 'w') as f:
        f.write(''.join(mu_qa_dict[k]['dev']))
print('Finished outputting mu_dev qa datasets...')


####################################################################################################
# check number of tokens in the longest decomp answer...
tokenizer = load_model(loadwhat='tokenizer_only')
tok_counts = []
for i, mu_sample in enumerate(mu_train):
    ans = mu_sample['decomp_ans_str']
    input_string = '<s>' + ans + '</s>'    
    ids = string_to_ids(input_string, tokenizer, verbose=False)
    tok_counts.append( len(ids) )
    if i % 1000 == 0:
        print(f'Processed: {i}')
tok_counts_np = np.array(tok_counts)
print(f"count:{len(tok_counts)} max:{tok_counts_np.max()} mean:{tok_counts_np.mean()}") # count:19938 max:112 mean:43.56816130003009
hittoklimit = np.where(tok_counts_np >= 512)
hittoklimit[0].shape  #

tok_counts = []
for i, mu_sample in enumerate(mu_dev):
    ans = mu_sample['decomp_ans_str']
    input_string = '<s>' + ans + '</s>'    
    ids = string_to_ids(input_string, tokenizer, verbose=False)
    tok_counts.append( len(ids) )
    if i % 1000 == 0:
        print(f'Processed: {i}')
tok_counts_np = np.array(tok_counts)
print(f"count:{len(tok_counts)} max:{tok_counts_np.max()} mean:{tok_counts_np.mean()}") # count:19938 max:112 mean:43.56816130003009
hittoklimit = np.where(tok_counts_np >= 512)
hittoklimit[0].shape  #


