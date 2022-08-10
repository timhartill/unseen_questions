#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 13:14:05 2022

@author: tim hartill

Read sentence-level annotations from original HPQA train/dev files and add to mdr-formatted HPQA train/dev files + qas_val file

Also converts the MDR_PROCESSED_CORPUS file into our format hpqa abstracts file.

Output format example:

{"question": "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?", 
 "answers": ["Chief of Protocol"], 
 "type": "bridge", 
 "pos_paras": [{"title": "Kiss and Tell (1945 film)", 
                "text": "Kiss and Tell is a 1945 American comedy film starring then 17-year-old Shirley Temple as Corliss Archer. In the film, two teenage girls cause their respective parents much concern when they start to become interested in boys. The parents' bickering about which girl is the worse influence causes more problems than it solves.", 
                "sentence_spans": [[0, 104], [104, 225], [225, 325]], 
                "sentence_labels": [0]}, 
               {"title": "Shirley Temple", "text": "Shirley Temple Black (April 23, 1928\u00a0\u2013 February 10, 2014) was an American actress, singer, dancer, businesswoman, and diplomat who was Hollywood's number one box-office draw as a child actress from 1935 to 1938. As an adult, she was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States.", "sentence_spans": [[0, 211], [211, 353]], "sentence_labels": [0, 1]}], 
 "neg_paras": [{"title": "A Kiss for Corliss", 
                "text": "A Kiss for Corliss is a 1949 American comedy film directed by Richard Wallace and written by Howard Dimsdale. It stars Shirley Temple in her final starring role as well as her final film appearance. It is a sequel to the 1945 film \"Kiss and Tell\". \"A Kiss for Corliss\" was retitled \"Almost a Bride\" before release and this title appears in the title sequence. The film was released on November 25, 1949, by United Artists."}, 
               {"title": "Kiss and Tell", "text": "Kiss and Tell or Kiss & Tell or Kiss n Tell may refer to:"}, {"title": "Secretary of State for Constitutional Affairs", "text": "The office of Secretary of State for Constitutional Affairs was a British Government position, created in 2003. Certain functions of the Lord Chancellor which related to the Lord Chancellor's Department were transferred to the Secretary of State. At a later date further functions were also transferred to the Secretary of State for Constitutional Affairs from the First Secretary of State, a position within the government held by the Deputy Prime Minister."}, {"title": "Joseph Kalite", "text": "Joseph Kalite (died 24 January 2014) was a Central African politician. As a government minister he either held the housing or health portfolio. Kalite, a Muslim, was reported to be killed by anti-balaka outside the Central Mosque in the capital Bangui during the Central African Republic conflict. He was killed with machetes on the day in Bangui after interim president Catherine Samba-Panza took power. At the time of the attack Kalite held no government position, nor did he under the S\u00e9l\u00e9ka rule. He was reported to have supported the rule of S\u00e9l\u00e9ka leader Michel Djotodia."}, {"title": "The Brave Archer and His Mate", "text": "The Brave Archer and His Mate, also known as The Brave Archer 4 and Mysterious Island, is a 1982 Hong Kong film adapted from Louis Cha's novels \"The Legend of the Condor Heroes\" and \"The Return of the Condor Heroes\". Together with \"Little Dragon Maiden\" (1983), \"The Brave Archer and His Mate\" is regarded as an unofficial sequel to the \"Brave Archer\" film trilogy (\"The Brave Archer\", \"The Brave Archer 2\" and \"The Brave Archer 3\")."}, {"title": "Little Dragon Maiden", "text": "Little Dragon Maiden, also known as The Brave Archer 5, is a 1983 Hong Kong film adapted from Louis Cha's novel \"The Return of the Condor Heroes\". \"Little Dragon Maiden\" and \"The Brave Archer and His Mate\" (1982) are seen as unofficial sequels to the \"Brave Archer\" film trilogy (\"The Brave Archer\", \"The Brave Archer 2\" and \"The Brave Archer 3\")."}, {"title": "The Brave Archer 3", "text": "The Brave Archer 3, also known as Blast of the Iron Palm, is a 1981 Hong Kong film adapted from Louis Cha's novel \"The Legend of the Condor Heroes\". The film was produced by the Shaw Brothers Studio and directed by Chang Cheh, starring Alexander Fu Sheng and Niu-niu in the lead roles. The film is the third part of a trilogy and was preceded by \"The Brave Archer\" (1977) and \"The Brave Archer 2\" (1978). The film has two unofficial sequels, \"The Brave Archer and His Mate\" (1982) and \"Little Dragon Maiden\" (1983), both of which were based on \"The Return of the Condor Heroes\". The theme song of the film, \"Say Cheung Kei\" (\u56db\u5f35\u6a5f), was composed by Chang Cheh, arranged by Joseph Koo and performed in Cantonese by Jenny Tseng."}, {"title": "The Brave Archer", "text": "The Brave Archer, also known as Kungfu Warlord, is a 1977 Hong Kong film adapted from Louis Cha's novel \"The Legend of the Condor Heroes\". The film was produced by the Shaw Brothers Studio and directed by Chang Cheh, starring Alexander Fu Sheng and Tanny Tien in the lead roles. The film is the first part of a trilogy and was followed by \"The Brave Archer 2\" (1978) and \"The Brave Archer 3\" (1981). The trilogy has two unofficial sequels, \"The Brave Archer and His Mate\" (1982) and \"Little Dragon Maiden\" (1983)."}, {"title": "Catherine Samba-Panza", "text": "Catherine Samba-Panza (born 26 June 1956) was interim President of the Central African Republic from 2014 to 2016. She was the first woman to hold the post of head of state in that country, as well as the eighth woman in Africa to do so. Prior to becoming head of state, she was Mayor of Bangui from 2013 to 2014."}, {"title": "The Brave Archer 2", "text": "The Brave Archer 2, also known as Kungfu Warlord 2, is a 1978 Hong Kong film adapted from Louis Cha's novel \"The Legend of the Condor Heroes\". The film was produced by the Shaw Brothers Studio and directed by Chang Cheh, starring Alexander Fu Sheng and Niu-niu in the lead roles. The film is the second part of a trilogy and was preceded by \"The Brave Archer\" (1977) and followed by \"The Brave Archer 3\" (1981). The trilogy has two unofficial sequels, \"The Brave Archer and His Mate\" (1982) and \"Little Dragon Maiden\" (1983)."}, {"title": "Charles Craft", "text": "Charles Craft (May 9, 1902 \u2013 September 19, 1968) was an English-born American film and television editor. Born in the county of Hampshire in England on May 9, 1902, Craft would enter the film industry in Hollywood in 1927. The first film he edited was the Universal Pictures silent film, \"Painting the Town\". Over the next 25 years, Craft would edit 90 feature-length films. In the early 1950s he would switch his focus to the small screen, his first show being \"Racket Squad\", from 1951\u201353, for which he was the main editor, editing 93 of the 98 episodes. He would work on several other series during the 1950s, including \"Meet Corliss Archer\" (1954), \"Science Fiction Theatre\" (1955\u201356), and \"Highway Patrol\" (1955\u201357). In the late 1950s and early 1960s he was one of the main editors on \"Sea Hunt\", starring Lloyd Bridges, editing over half of the episodes. His final film work would be editing \"Flipper's New Adventure\" (1964, the sequel to 1963's \"Flipper\". When the film was made into a television series, Craft would begin the editing duties on that show, editing the first 28 episodes before he retired in 1966. Craft died on September 19, 1968 in Los Angeles, California."}, {"title": "Lord High Treasurer", "text": "The post of Lord High Treasurer or Lord Treasurer was an English government position and has been a British government position since the Acts of Union of 1707. A holder of the post would be the third-highest-ranked Great Officer of State, below the Lord High Steward and the Lord High Chancellor."}, {"title": "Village accountant", "text": "The Village Accountant (variously known as \"Patwari\", \"Talati\", \"Patel\", \"Karnam\", \"Adhikari\", \"Shanbogaru\",\"Patnaik\" etc.) is an administrative government position found in rural parts of the Indian sub-continent. The office and the officeholder are called the \"patwari\" in Telangana, Bengal, North India and in Pakistan while in Sindh it is called \"tapedar\". The position is known as the \"karnam\" in Andhra Pradesh, \"patnaik\" in Orissa or \"adhikari\" in Tamil Nadu, while it is commonly known as the \"talati\" in Karnataka, Gujarat and Maharashtra. The position was known as the \"kulkarni\" in Northern Karnataka and Maharashtra. The position was known as the \"shanbogaru\" in South Karnataka."}, {"title": "Under-Secretary of State for War", "text": "The position of Under-Secretary of State for War was a British government position, first applied to Evan Nepean (appointed in 1794). In 1801 the offices for War and the Colonies were merged and the post became that of Under-Secretary of State for War and the Colonies. The position was re-instated in 1854 and remained until 1947, when it was combined with that of Financial Secretary to the War Office. In 1964 the War Office, Admiralty and Air Ministry were merged to form the Ministry of Defence, and the post was abolished."}, {"title": "Yeonguijeong", "text": "Yeonguijeong (] ) was a title created in 1400, during the Joseon Dynasty of Korea (1392-1910) and given to the Chief State Councillor as the highest government position of \"Uijeongbu\" (State Council). Existing for over 500 years, the function was handed over in 1895 during the Gabo Reform to the newly formed position of Prime Minister of Korea. Only one official at a time was appointed to the position and though was generally called \"Yeongsang\", was also referred to as \"Sangsang\", \"Sugyu\" or \"Wonbo\". Although, the title of Yeonguijeong was defined as the highest post in charge of every state affairs by law, its practical functions changed drastically depending on the particular King and whether that King's power was strong or weak."}, {"title": "Janet Waldo", "text": "Janet Marie Waldo (February 4, 1920 \u2013 June 12, 2016) was an American radio and voice actress. She is best known in animation for voicing Judy Jetson, Nancy in \"Shazzan\", Penelope Pitstop, and Josie in \"Josie and the Pussycats\", and on radio as the title character in \"Meet Corliss Archer\"."}, {"title": "Meet Corliss Archer (TV series)", "text": "Meet Corliss Archer is an American television sitcom that aired on CBS (July 13, 1951 - August 10, 1951) and in syndication via the Ziv Company from April to December 1954. The program was an adaptation of the radio series of the same name, which was based on a series of short stories by F. Hugh Herbert."}, {"title": "Centennial Exposition", "text": "The Centennial International Exhibition of 1876, the first official World's Fair in the United States, was held in Philadelphia, Pennsylvania, from May 10 to November 10, 1876, to celebrate the 100th anniversary of the signing of the Declaration of Independence in Philadelphia. Officially named the International Exhibition of Arts, Manufactures and Products of the Soil and Mine, it was held in Fairmount Park along the Schuylkill River on fairgrounds designed by Herman J. Schwarzmann. Nearly 10 million visitors attended the exhibition and thirty-seven countries participated in it."}, {"title": "Meet Corliss Archer", "text": "Meet Corliss Archer, a program from radio's Golden Age, ran from January 7, 1943 to September 30, 1956. Although it was CBS's answer to NBC's popular \"A Date with Judy\", it was also broadcast by NBC in 1948 as a summer replacement for \"The Bob Hope Show\". From October 3, 1952 to June 26, 1953, it aired on ABC, finally returning to CBS. Despite the program's long run, fewer than 24 episodes are known to exist."}], 
 "bridge": "Shirley Temple", 
 "_id": "5a8c7595554299585d9e36b6"}

Notes:
Added output of UQA formatted files, both open domain form and q + c -> a where c mixes gold and distractor paras, some with random sents removed. paras randomly preceded with 'title': or not

"""

import os
import json
import random
from html import unescape

from transformers import AutoTokenizer

import eval_metrics
import utils
from text_processing import create_sentence_spans, split_into_sentences

HPQA_DEV = '/home/thar011/data/hpqa/hotpot_dev_fullwiki_v1.json'
HPQA_TRAIN = '/home/thar011/data/hpqa/hotpot_train_v1.1.json'

MDR_DEV = '/large_data/thar011/out/mdr/encoded_corpora/hotpot/hotpot_dev_with_neg_v0.json'
MDR_TRAIN = '/large_data/thar011/out/mdr/encoded_corpora/hotpot/hotpot_train_with_neg_v0.json'
MDR_QAS_VAL = '/large_data/thar011/out/mdr/encoded_corpora/hotpot/hotpot_qas_val.json'
MDR_PROCESSED_CORPUS = '/large_data/thar011/out/mdr/encoded_corpora/hotpot/wiki_id2doc_from_mdr_with_sent_splits.json'  # was '/data/thar011/gitrepos/compgen_mdr/data/hotpot_index/wiki_id2doc.json'

MDR_UPDATED_DEV = '/large_data/thar011/out/mdr/encoded_corpora/hotpot/hotpot_dev_with_neg_v0_sentannots.jsonl'
MDR_UPDATED_TRAIN = '/large_data/thar011/out/mdr/encoded_corpora/hotpot/hotpot_train_with_neg_v0_sentannots.jsonl'
MDR_UPDATED_QAS_VAL = '/large_data/thar011/out/mdr/encoded_corpora/hotpot/hotpot_qas_val_with_spfacts.jsonl'
MDR_UPDATED_CORPUS = '/large_data/thar011/out/mdr/encoded_corpora/hotpot/hpqa_abstracts_with_sent_spans.jsonl'

UQA_DIR = '/data/thar011/data/unifiedqa/'

addspecialtoksdict = eval_metrics.special_tokens_dict  # test tokenization length with ind. digit tokenization...
tokenizer = utils.load_model(model_name='facebook/bart-large', loadwhat='tokenizer_only', special_tokens_dict=addspecialtoksdict)        

max_toks = 512
added_bits = len(tokenizer.tokenize('<s>. \\n</s>'))
max_toks = max_toks - added_bits
print(f"Max num tokens for text after allowing for BOS, EOS etc: {max_toks}")



hpqa_dev = json.load(open(HPQA_DEV))        # 7405 dict_keys(['_id', 'answer', 'question', 'supporting_facts', 'context', 'type', 'level']) 
hpqa_train = json.load(open(HPQA_TRAIN))    # 90447 #'supporting_facts' = [ [title, sentence_idx], ..]

mdr_dev = utils.load_jsonl(MDR_DEV)         # 7405  dict_keys(['question', 'answers', 'type', 'pos_paras', 'neg_paras', '_id'])
mdr_qas_val = utils.load_jsonl(MDR_QAS_VAL) # 7405  dict_keys(['question', '_id', 'answer', 'sp', 'type'])
mdr_train = utils.load_jsonl(MDR_TRAIN)     # 90447
mdr_corpus = json.load(open(MDR_PROCESSED_CORPUS)) # 5233329  MDR processed file has the sentence mappings, easier than doing raw processing on hpqa wiki dump..

# create abstracts in our format:
mdr_abstracts_out = [{'title': v['title'], 'text': v['text'], 
                      'sentence_spans': create_sentence_spans(v['sents'])} for v in mdr_corpus.values() if v['text'].strip() != ''] # 5233235 strips blanks
utils.saveas_jsonl(mdr_abstracts_out, MDR_UPDATED_CORPUS)
del mdr_abstracts_out


def build_title_dict(mdr_corpus):
    """ Convert MDR processed corpus file from:
        {'idx':{'title': 'title', 'text':'abstract text', 'sents': ['Sentence 1.', ' Sentence 2.', ...]} }
    """
    corpus_sents = {}
    dup_titles = []
    for idx in mdr_corpus:
        spans = create_sentence_spans(mdr_corpus[idx]['sents'])
        title = mdr_corpus[idx]['title']
        if corpus_sents.get(unescape(title)) is not None:
            print(f"Duplicate title: {title} (unescaped: {unescape(title)})")
            dup_titles.append(title)
        corpus_sents[unescape(title)] = {'sentence_spans':spans, 'text': mdr_corpus[idx]['text']}
    print(f"Duplicate titles: {len(dup_titles)}")  #0
    return corpus_sents, dup_titles

hpqa_sentence_spans, dup_titles = build_title_dict(mdr_corpus)

# [] and true for mdr_train para 1 and mdr_dev both paras also...so training para text lines up with corpus text..
#[m for m in mdr_train if m['pos_paras'][0]['text'] != hpqa_sentence_spans[unescape(m['pos_paras'][0]['title'])]['text'] ]


def aggregate_sent_annots(supporting_facts):
    """ Aggregate supporting fars from eg [['Allie Goertz', 0], ['Allie Goertz', 1], ['Allie Goertz', 2], ['Milhouse Van Houten', 0]]
    to {'Allie Goertz': [0,1,2], 'Milhouse Van Houten': [0]}
    """
    label_dict = {}
    for t, s in supporting_facts: 
        title_unescaped = unescape(t)
        if label_dict.get(title_unescaped) is None:
            label_dict[title_unescaped] = []
        label_dict[title_unescaped].append(s)
    for t in label_dict:
        label_dict[t].sort()    
    return label_dict
        


def add_span_and_sent_annot(mdr_split, hpqa_split, sentence_spans):
    """ Add keys for sentence_spans and sentence annotations
    Note: mdr and hpqa files are in the same order...
    """
    for i, s in enumerate(mdr_split):
        sent_labels = aggregate_sent_annots(hpqa_split[i]['supporting_facts'])
        for para in s['pos_paras']:
            title_unescaped = unescape(para['title'])  # a few titles are actually escaped..
            spans = sentence_spans[title_unescaped]['sentence_spans']
            para['sentence_spans'] = spans
            para['sentence_labels'] = sent_labels[title_unescaped]
        if i % 10000 == 0:
            print(f"Processed: {i}")
    return


def add_spfacts(mdr_qas_val, mdr_dev):
    """ Add sp_facts key to qas_val of form: [ [title1, pos sent idx1], [title1, pos sent idx2], [title2, pos sent idx1], ... ]
    Note: mdr_qas_val and mdr_dev files are in the same order...
    """
    for i, s in enumerate(mdr_dev):
        sp_facts = []
        for para in s['pos_paras']:
            for slabel in para['sentence_labels']:
                sp_facts.append( [para['title'], slabel] )  
            mdr_qas_val[i]['sp_facts'] = sp_facts
            if para['title'] not in mdr_qas_val[i]['sp']:  # check title escape/unescape matches - they all do
                print(f"{i} Title mismatch: mdr_dev:{para['title']} mdr_qas_val:{mdr_qas_val[i]['sp']}")
    return
    

def check_sentence_annots(split):
    """ check sentence annotations are all valid
    """
    errlist = []
    for i, s in enumerate(split):
        for j, para in enumerate(s['pos_paras']):
            if len(para['sentence_labels']) == 0:
                print(f"sample:{i} pospara: {j} Zero length sentence label!")
                errlist.append([i,j,"zero len label"])
            for sent_idx in para['sentence_labels']:
                if sent_idx >= len(para['sentence_spans']):
                    print(f"sample:{i} pospara: {j} sent idx > # sentences!")
                    errlist.append([i,j,"idx > # sents"])
                else:
                    start, end = para['sentence_spans'][sent_idx]
                    if start < 0 or len(para['text']) == 0:
                        print(f"sample:{i} pospara: {j} invalid start")
                        errlist.append([i,j,"invalid start or zero len text"])
                    if end > len(para['text']):    
                        print(f"sample:{i} pospara: {j} invalid end")
                        errlist.append([i,j,"invalid end"])
    return errlist
    

add_span_and_sent_annot(mdr_dev, hpqa_dev, hpqa_sentence_spans)
add_span_and_sent_annot(mdr_train, hpqa_train, hpqa_sentence_spans)

utils.saveas_jsonl(mdr_dev, MDR_UPDATED_DEV)
utils.saveas_jsonl(mdr_train, MDR_UPDATED_TRAIN)

add_spfacts(mdr_qas_val, mdr_dev)
utils.saveas_jsonl(mdr_qas_val, MDR_UPDATED_QAS_VAL)


################
# Create UQA standard formatted hpqa files
################
mdr_dev = utils.load_jsonl(MDR_UPDATED_DEV)     #7405 
mdr_train = utils.load_jsonl(MDR_UPDATED_TRAIN) #90447  dict_keys(['question', 'answers', 'type', 'pos_paras', 'neg_paras', '_id' [, 'bridge']])


def make_samples(split, tokenizer, max_toks=507, include_title_prob=0.5, include_all_sent_prob=0.5, 
                 keep_pos_sent_prob=0.5, keep_neg_sent_prob=0.6):
    """ Create standard UQA formatted samples from "mdr" format dict_keys(['question', 'answers', 'type', 'pos_paras', 'neg_paras', '_id' [, 'bridge']])
        with q + paras per doc packed in to roughly max_toks toks.
    
        Note: Short docs will be less than 512 toks. We dont pack more in to these to preserve diversity. 
              Also some may end up slightly over max_toks.
    """
    out_list = []
    for i, s in enumerate(split):
        tok_count = len(tokenizer.tokenize(s['question']))
        
        para_list = []  #list so we can shuffle
        for para in s['pos_paras']:
            text = ''
            if random.random() < include_title_prob:
                text += unescape(para['title']).strip() + ': '
            if random.random() < include_all_sent_prob or len(para['sentence_spans']) <= 1:  # include full para text
                text += para['text'].strip()
                if text[-1] not in ['.', '!', '?', ':', ';']:
                    text += '.'
                text += ' '
            else:                                                                            # include gold + partial other sentences
                for j, (start, end) in enumerate(para['sentence_spans']):
                    if j in para['sentence_labels'] or (random.random() < keep_pos_sent_prob):
                        text += para['text'][start:end].strip()
                        if text[-1] not in ['.', '!', '?', ':', ';']:
                            text += '.'
                        text += ' '
            tok_count += len(tokenizer.tokenize(text))
            para_list.append(text.strip())
            
        for para in s['neg_paras']:
            text = ''
            if random.random() < include_title_prob:
                text += unescape(para['title']).strip() + ': '
            if random.random() < include_all_sent_prob:  # include full para text
                text += para['text'].strip()
                if text[-1] not in ['.', '!', '?', ':', ';']:
                    text += '.'
                text += ' '
            else:                                        # include subset of para sentences
                sentence_spans = create_sentence_spans(split_into_sentences(para['text']))
                if len(sentence_spans) > 1:
                    for j, (start, end) in enumerate(sentence_spans):
                        if random.random() < keep_neg_sent_prob:
                            text += para['text'][start:end].strip()
                            if text[-1] not in ['.', '!', '?', ':', ';']:
                                text += '.'
                            text += ' '
                else:
                    text += para['text'].strip()
                    if text[-1] not in ['.', '!', '?', ':', ';']:
                        text += '.'
                    text += ' '
            para_toks = tokenizer.tokenize(text)            
            if tok_count + len(para_toks) > max_toks:
                excess = max_toks - (tok_count+len(para_toks)+1)
                if excess > 25:
                    para_toks = para_toks[:excess]
                    para_truncated = tokenizer.decode(tokenizer.convert_tokens_to_ids(para_toks)) + '...'
                    para_list.append(para_truncated.strip())
                break
            else:
                tok_count += len(para_toks) + 1
                para_list.append(text.strip())
        random.shuffle(para_list)
        context = ' '.join(para_list)
        if type(s['answers']) == list and len(s['answers']) == 1:
            answer = str(s['answers'][0])
        else: 
            answer = s['answers']
        out_list.append( utils.create_uqa_example(s['question'], context, answer, append_q_char='?') )
        if i % 1000 == 0:
            print(f"Loaded {i} samples of {len(split)}...")
    return out_list


random.seed(42)
dev_out = make_samples(mdr_dev, tokenizer, max_toks, include_title_prob=0.5, include_all_sent_prob=0.5)
out_dir = os.path.join(UQA_DIR, "hpqa_hard")
print(f'Outputting to {out_dir}')
os.makedirs(out_dir, exist_ok=True)
outfile = os.path.join(out_dir, 'dev.tsv')
print(f"Outputting: {outfile}")
with open(outfile, 'w') as f:
    f.write(''.join(dev_out))
    
train_out = make_samples(mdr_train, tokenizer, max_toks, include_title_prob=0.5, include_all_sent_prob=0.5)
outfile = os.path.join(out_dir, 'train.tsv')
print(f"Outputting: {outfile}")
with open(outfile, 'w') as f:
    f.write(''.join(train_out))
print('Finished!')



# Below is error checking stuff

errlist = check_sentence_annots(mdr_dev) # sample:5059 pospara: 0 sent idx > # sentences! Error is in hpqa sentence annot
errlist = check_sentence_annots(mdr_train)  #Errors all seem to be in the hpqa sentence annots
#sample:514 pospara: 1 sent idx > # sentences!
#sample:8332 pospara: 0 sent idx > # sentences!
#sample:9548 pospara: 1 sent idx > # sentences!
#sample:13415 pospara: 0 sent idx > # sentences!
#sample:20594 pospara: 0 sent idx > # sentences!
#sample:22896 pospara: 0 sent idx > # sentences!
#sample:27436 pospara: 0 sent idx > # sentences!
#sample:37004 pospara: 1 sent idx > # sentences!
#sample:38579 pospara: 0 sent idx > # sentences!
#sample:41267 pospara: 0 sent idx > # sentences!
#sample:45705 pospara: 1 sent idx > # sentences!
#sample:49355 pospara: 0 sent idx > # sentences!
#sample:50651 pospara: 1 sent idx > # sentences!
#sample:52080 pospara: 0 sent idx > # sentences!
#sample:60885 pospara: 1 sent idx > # sentences!
#sample:67475 pospara: 0 sent idx > # sentences!
#sample:77109 pospara: 0 sent idx > # sentences!
#sample:85934 pospara: 0 sent idx > # sentences!
#sample:86118 pospara: 1 sent idx > # sentences!
#sample:86193 pospara: 1 sent idx > # sentences!
#sample:88641 pospara: 0 sent idx > # sentences!
#sample:89961 pospara: 0 sent idx > # sentences!




