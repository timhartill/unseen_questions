#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:55:31 2022

@author: tim hartill

convert hover data to standard input format:

{"question": "[CLAIM] What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?", 
 "answers": ["Chief of Protocol"], [LABEL] 
 "src": 'fever',
 "type": "bridge",  [fever] 
 "pos_paras": [{"title": "Kiss and Tell (1945 film)", 
                "text": "Kiss and Tell is a 1945 American comedy film starring then 17-year-old Shirley Temple as Corliss Archer. In the film, two teenage girls cause their respective parents much concern when they start to become interested in boys. The parents' bickering about which girl is the worse influence causes more problems than it solves.", 
                "sentence_spans": [[0, 104], [104, 225], [225, 325]], 
                "sentence_labels": [0]}, 
               {"title": "Shirley Temple", "text": "Shirley Temple Black (April 23, 1928\u00a0\u2013 February 10, 2014) was an American actress, singer, dancer, businesswoman, and diplomat who was Hollywood's number one box-office draw as a child actress from 1935 to 1938. As an adult, she was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States.", "sentence_spans": [[0, 211], [211, 353]], "sentence_labels": [0, 1]}], 
 "neg_paras": [ [N/A] {"title": "A Kiss for Corliss", 
                "text": "A Kiss for Corliss is a 1949 American comedy film directed by Richard Wallace and written by Howard Dimsdale. It stars Shirley Temple in her final starring role as well as her final film appearance. It is a sequel to the 1945 film \"Kiss and Tell\". \"A Kiss for Corliss\" was retitled \"Almost a Bride\" before release and this title appears in the title sequence. The film was released on November 25, 1949, by United Artists."}, 
               {"title": "Kiss and Tell", "text": "Kiss and Tell or Kiss & Tell or Kiss n Tell may refer to:"}, {"title": "Secretary of State for Constitutional Affairs", "text": "The office of Secretary of State for Constitutional Affairs was a British Government position, created in 2003. Certain functions of the Lord Chancellor which related to the Lord Chancellor's Department were transferred to the Secretary of State. At a later date further functions were also transferred to the Secretary of State for Constitutional Affairs from the First Secretary of State, a position within the government held by the Deputy Prime Minister."}, {"title": "Joseph Kalite", "text": "Joseph Kalite (died 24 January 2014) was a Central African politician. As a government minister he either held the housing or health portfolio. Kalite, a Muslim, was reported to be killed by anti-balaka outside the Central Mosque in the capital Bangui during the Central African Republic conflict. He was killed with machetes on the day in Bangui after interim president Catherine Samba-Panza took power. At the time of the attack Kalite held no government position, nor did he under the S\u00e9l\u00e9ka rule. He was reported to have supported the rule of S\u00e9l\u00e9ka leader Michel Djotodia."}, {"title": "The Brave Archer and His Mate", "text": "The Brave Archer and His Mate, also known as The Brave Archer 4 and Mysterious Island, is a 1982 Hong Kong film adapted from Louis Cha's novels \"The Legend of the Condor Heroes\" and \"The Return of the Condor Heroes\". Together with \"Little Dragon Maiden\" (1983), \"The Brave Archer and His Mate\" is regarded as an unofficial sequel to the \"Brave Archer\" film trilogy (\"The Brave Archer\", \"The Brave Archer 2\" and \"The Brave Archer 3\")."}, {"title": "Little Dragon Maiden", "text": "Little Dragon Maiden, also known as The Brave Archer 5, is a 1983 Hong Kong film adapted from Louis Cha's novel \"The Return of the Condor Heroes\". \"Little Dragon Maiden\" and \"The Brave Archer and His Mate\" (1982) are seen as unofficial sequels to the \"Brave Archer\" film trilogy (\"The Brave Archer\", \"The Brave Archer 2\" and \"The Brave Archer 3\")."}, {"title": "The Brave Archer 3", "text": "The Brave Archer 3, also known as Blast of the Iron Palm, is a 1981 Hong Kong film adapted from Louis Cha's novel \"The Legend of the Condor Heroes\". The film was produced by the Shaw Brothers Studio and directed by Chang Cheh, starring Alexander Fu Sheng and Niu-niu in the lead roles. The film is the third part of a trilogy and was preceded by \"The Brave Archer\" (1977) and \"The Brave Archer 2\" (1978). The film has two unofficial sequels, \"The Brave Archer and His Mate\" (1982) and \"Little Dragon Maiden\" (1983), both of which were based on \"The Return of the Condor Heroes\". The theme song of the film, \"Say Cheung Kei\" (\u56db\u5f35\u6a5f), was composed by Chang Cheh, arranged by Joseph Koo and performed in Cantonese by Jenny Tseng."}, {"title": "The Brave Archer", "text": "The Brave Archer, also known as Kungfu Warlord, is a 1977 Hong Kong film adapted from Louis Cha's novel \"The Legend of the Condor Heroes\". The film was produced by the Shaw Brothers Studio and directed by Chang Cheh, starring Alexander Fu Sheng and Tanny Tien in the lead roles. The film is the first part of a trilogy and was followed by \"The Brave Archer 2\" (1978) and \"The Brave Archer 3\" (1981). The trilogy has two unofficial sequels, \"The Brave Archer and His Mate\" (1982) and \"Little Dragon Maiden\" (1983)."}, {"title": "Catherine Samba-Panza", "text": "Catherine Samba-Panza (born 26 June 1956) was interim President of the Central African Republic from 2014 to 2016. She was the first woman to hold the post of head of state in that country, as well as the eighth woman in Africa to do so. Prior to becoming head of state, she was Mayor of Bangui from 2013 to 2014."}, {"title": "The Brave Archer 2", "text": "The Brave Archer 2, also known as Kungfu Warlord 2, is a 1978 Hong Kong film adapted from Louis Cha's novel \"The Legend of the Condor Heroes\". The film was produced by the Shaw Brothers Studio and directed by Chang Cheh, starring Alexander Fu Sheng and Niu-niu in the lead roles. The film is the second part of a trilogy and was preceded by \"The Brave Archer\" (1977) and followed by \"The Brave Archer 3\" (1981). The trilogy has two unofficial sequels, \"The Brave Archer and His Mate\" (1982) and \"Little Dragon Maiden\" (1983)."}, {"title": "Charles Craft", "text": "Charles Craft (May 9, 1902 \u2013 September 19, 1968) was an English-born American film and television editor. Born in the county of Hampshire in England on May 9, 1902, Craft would enter the film industry in Hollywood in 1927. The first film he edited was the Universal Pictures silent film, \"Painting the Town\". Over the next 25 years, Craft would edit 90 feature-length films. In the early 1950s he would switch his focus to the small screen, his first show being \"Racket Squad\", from 1951\u201353, for which he was the main editor, editing 93 of the 98 episodes. He would work on several other series during the 1950s, including \"Meet Corliss Archer\" (1954), \"Science Fiction Theatre\" (1955\u201356), and \"Highway Patrol\" (1955\u201357). In the late 1950s and early 1960s he was one of the main editors on \"Sea Hunt\", starring Lloyd Bridges, editing over half of the episodes. His final film work would be editing \"Flipper's New Adventure\" (1964, the sequel to 1963's \"Flipper\". When the film was made into a television series, Craft would begin the editing duties on that show, editing the first 28 episodes before he retired in 1966. Craft died on September 19, 1968 in Los Angeles, California."}, {"title": "Lord High Treasurer", "text": "The post of Lord High Treasurer or Lord Treasurer was an English government position and has been a British government position since the Acts of Union of 1707. A holder of the post would be the third-highest-ranked Great Officer of State, below the Lord High Steward and the Lord High Chancellor."}, {"title": "Village accountant", "text": "The Village Accountant (variously known as \"Patwari\", \"Talati\", \"Patel\", \"Karnam\", \"Adhikari\", \"Shanbogaru\",\"Patnaik\" etc.) is an administrative government position found in rural parts of the Indian sub-continent. The office and the officeholder are called the \"patwari\" in Telangana, Bengal, North India and in Pakistan while in Sindh it is called \"tapedar\". The position is known as the \"karnam\" in Andhra Pradesh, \"patnaik\" in Orissa or \"adhikari\" in Tamil Nadu, while it is commonly known as the \"talati\" in Karnataka, Gujarat and Maharashtra. The position was known as the \"kulkarni\" in Northern Karnataka and Maharashtra. The position was known as the \"shanbogaru\" in South Karnataka."}, {"title": "Under-Secretary of State for War", "text": "The position of Under-Secretary of State for War was a British government position, first applied to Evan Nepean (appointed in 1794). In 1801 the offices for War and the Colonies were merged and the post became that of Under-Secretary of State for War and the Colonies. The position was re-instated in 1854 and remained until 1947, when it was combined with that of Financial Secretary to the War Office. In 1964 the War Office, Admiralty and Air Ministry were merged to form the Ministry of Defence, and the post was abolished."}, {"title": "Yeonguijeong", "text": "Yeonguijeong (] ) was a title created in 1400, during the Joseon Dynasty of Korea (1392-1910) and given to the Chief State Councillor as the highest government position of \"Uijeongbu\" (State Council). Existing for over 500 years, the function was handed over in 1895 during the Gabo Reform to the newly formed position of Prime Minister of Korea. Only one official at a time was appointed to the position and though was generally called \"Yeongsang\", was also referred to as \"Sangsang\", \"Sugyu\" or \"Wonbo\". Although, the title of Yeonguijeong was defined as the highest post in charge of every state affairs by law, its practical functions changed drastically depending on the particular King and whether that King's power was strong or weak."}, {"title": "Janet Waldo", "text": "Janet Marie Waldo (February 4, 1920 \u2013 June 12, 2016) was an American radio and voice actress. She is best known in animation for voicing Judy Jetson, Nancy in \"Shazzan\", Penelope Pitstop, and Josie in \"Josie and the Pussycats\", and on radio as the title character in \"Meet Corliss Archer\"."}, {"title": "Meet Corliss Archer (TV series)", "text": "Meet Corliss Archer is an American television sitcom that aired on CBS (July 13, 1951 - August 10, 1951) and in syndication via the Ziv Company from April to December 1954. The program was an adaptation of the radio series of the same name, which was based on a series of short stories by F. Hugh Herbert."}, {"title": "Centennial Exposition", "text": "The Centennial International Exhibition of 1876, the first official World's Fair in the United States, was held in Philadelphia, Pennsylvania, from May 10 to November 10, 1876, to celebrate the 100th anniversary of the signing of the Declaration of Independence in Philadelphia. Officially named the International Exhibition of Arts, Manufactures and Products of the Soil and Mine, it was held in Fairmount Park along the Schuylkill River on fairgrounds designed by Herman J. Schwarzmann. Nearly 10 million visitors attended the exhibition and thirty-seven countries participated in it."}, {"title": "Meet Corliss Archer", "text": "Meet Corliss Archer, a program from radio's Golden Age, ran from January 7, 1943 to September 30, 1956. Although it was CBS's answer to NBC's popular \"A Date with Judy\", it was also broadcast by NBC in 1948 as a summer replacement for \"The Bob Hope Show\". From October 3, 1952 to June 26, 1953, it aired on ABC, finally returning to CBS. Despite the program's long run, fewer than 24 episodes are known to exist."}], 
 "bridge": ["Shirley Temple"], [all paras]
 "_id": "5a8c7595554299585d9e36b6"}  [F + fever id]
}
"""

import os
import json
import random
from html import unescape

import utils
from text_processing import normalize_unicode, convert_brc, replace_chars, create_sentence_spans, strip_accents

DEV = '/home/thar011/data/baleen_downloads/hover/dev/qas.json'
TRAIN = '/home/thar011/data/baleen_downloads/hover/train/qas.json'
TRAIN_PARA_SEQ = '/home/thar011/data/baleen_downloads/hover/baleen_hover_training_order2.json'
#PROCESSED_CORPUS_DIR = '/large_data/thar011/out/mdr/encoded_corpora/hotpot/wiki_id2doc_from_mdr_with_sent_splits.json'
PROCESSED_CORPUS_DIR_BALEEN = '/home/thar011/data/baleen_downloads/wiki.abstracts.2017/collection.json'
BQA_CORPUS = '/home/thar011/data/beerqa/enwiki-20200801-pages-articles-compgen-withmerges.jsonl'
BQA_TITLE_SAVE = '/home/thar011/data/beerqa/enwiki-20200801-titledict-compgen.json'

UPDATED_DEV = '/home/thar011/data/baleen_downloads/hover/hover_dev_with_neg_and_sent_annots.jsonl'
UPDATED_TRAIN = '/home/thar011/data/baleen_downloads/hover/hover_train_with_neg_and_sent_annots.jsonl'

QAS_VAL_FILE_OUT = '/home/thar011/data/baleen_downloads/hover/hover_qas_val_with_spfacts.jsonl'


#hpqa_corpus = json.load(open(PROCESSED_CORPUS_DIR))             # 5233329
baleen_corpus = utils.load_jsonl(PROCESSED_CORPUS_DIR_BALEEN)   # 5233330 but 1st row is {'pid': 0, 'title': '', 'text': []}
hover_dev = utils.load_jsonl(DEV) # 4000
hover_train = utils.load_jsonl(TRAIN) # 18171
hover_seq = json.load(open(TRAIN_PARA_SEQ))  #18171

baleen_dict = {w['title']:{'sents': w['text'], 'pid': w['pid']} for w in baleen_corpus if w['title'] != ''}
pid_to_title = {w['pid']: w['title'] for w in baleen_corpus if w['title'] != ''}

#for k in hpqa_corpus.keys():
#    h = hpqa_corpus[k]
#    w = baleen_dict[ h['title'] ]
#    if h['sents'] != w['sents']:
#        print(f"sents are different: {k} {h['title']}")  # No differences found..
#        break

for t in baleen_dict.keys():
    baleen_dict[t]['sentence_spans'] = create_sentence_spans(baleen_dict[t]['sents'])
    

def recursive_replace(nested_list, pid_to_title):
    """ Replace eg [[[670219, 811545, 1513707], 829606], 
                     [[670219, 1513707], 811545], 
                     [[], 670219],
                     [[811545, 670219, 1513707], 829606],
                     [[], 1513707]]
        with: 
            [[['Kristian Zahrtmann', 'Ossian Elgström', 'Peder Severin Krøyer'], 'Exlex'],
             [['Kristian Zahrtmann', 'Peder Severin Krøyer'], 'Ossian Elgström'],
             [[], 'Kristian Zahrtmann'],
             [['Ossian Elgström', 'Kristian Zahrtmann', 'Peder Severin Krøyer'], 'Exlex'],
             [[], 'Peder Severin Krøyer']]
    """
    if type(nested_list) == list:
        for i, element in enumerate(nested_list):
            if type(element) == list:
                nested_list[i] = recursive_replace(element, pid_to_title)
            else:
                nested_list[i] = unescape(pid_to_title[element])
    else:
        return unescape(pid_to_title[nested_list])
    return nested_list        


def remove_dups(nested_list):
    """ Sometimes there are dups in different orders eg: 
        [[[670219, 811545, 1513707], 829606], 
        [[670219, 1513707], 811545], 
        [[], 670219],
        [[811545, 670219, 1513707], 829606],
        [[], 1513707]]
    """
    new_list = []
    set_list = []
    for seq in nested_list:
        seq_flat = set(utils.flatten(seq))
        if seq_flat not in set_list:
            set_list.append(seq_flat)
            new_list.append(seq)
    return new_list
        


def calc_dependencies(nested_list, verbose=False):
    """ Calculate ordering(s) based on dependencies in the list. 
    eg qid 55 [  [[4775329, 2790082], 5038273],
                 [[], 2790082],
                 [[4775329, 2790082], 3395172],
                 [[4775329, 2790082, 3395172], 5038273],
                 [[], 4775329]]
    
    should resolve to allowable question + para sequence paths:
         [[4775329, 2790082], [3395172, 5038273]] 
         where [4775329, 2790082] means either of these can be first and the other second ie both reachable directly from question
         and [3395172, 5038273] means similarly use in either order but must come after both [4775329, 2790082] in query order
    """
    seqs = remove_dups( nested_list )
    seqs_sorted = sorted(seqs, key=lambda x: len(utils.flatten(x)), reverse=False)
    new_seqs = []
    curr_seq = []
    seen_paras = set()
    if verbose:
        print(f"Sequences: {seqs_sorted}")
    prior_length = -1
    for seq in seqs_sorted:
        curr_length = len(utils.flatten(seq))
        if curr_length > prior_length:
            prior_length = curr_length
            if curr_seq != []:
                if verbose:
                    print(f"Adding new seq: {curr_seq} to output and starting new level")
                new_seqs.append( curr_seq )
                curr_seq = []
        if verbose:
            print(f"Processing: {seq} length:{curr_length}")
        
        if seq[1] in seen_paras:  # ignore since already processed this para 
            if verbose: 
                print(f"Ignoring {seq[1]} since already processed")
            continue
        else:
            if seq[1] not in curr_seq:  # add to current level if not aready there
                curr_seq.append( seq[1] )
                seen_paras.add( seq[1] )
            elif verbose:
                print(f"Ignoring since {seq[1]} already in currseq:{curr_seq}")
        for p in seq[0]:
            if p not in seen_paras:  # Never happens but no harm in checking..
                print(f"NOTE: {p} not in seen:{seen_paras} adding to newseqs at prior level")
                new_seqs[-1].append( p )
                seen_paras.append( p )
    if curr_seq != []:
        if verbose:
            print(f"Adding new seq: {curr_seq} to output and finishing..")
        new_seqs.append( curr_seq )
    return new_seqs               



#for k in hover_seq:
#    new_seqs = calc_dependencies(hover_seq[k], verbose=False)
#    if len(utils.flatten(new_seqs)) != hover_train[int(k)]['num_hops']:
#        print(f"{k} Num hops != num paras in seq")
#        break
#    seqs = remove_dups( hover_seq[ k ] )
#    seqs_sorted = sorted(seqs, key=lambda x: len(utils.flatten(x)), reverse=True)
#    if seqs_sorted[0][0] != [] and len( utils.flatten(seqs_sorted[0]) ) == len( utils.flatten(seqs_sorted[1]) ):
#        print(f"same length 0 & 1: {k} {seqs_sorted}")
#        break
#    if len(seqs_sorted[0]) != 2:
#        print(f'1st item not 2: {seqs_sorted}')
#        break
#    for seqs in hover_seq[k]:
#        if type(seqs[-1]) != int:
#            print(f"id:[k] last element not a list: {hover_seq[k]}")
#            break
#    if type(seqs[-1]) != int:
#        break


    

def aggregate_evidence(evidence_list, corpus_dict, pid_to_title):
    """ Convert Hover/Baleen support facts/evidence into standard format
    ie evidence_list: [[670219, 0], [670219, 1], [1513707, 1], [811545, 2]]
    ->  [{"title": "Kiss and Tell (1945 film)", 
         "text": "Kiss and Tell is a 1945 American comedy film starring then 17-year-old Shirley Temple as Corliss Archer. In the film, two teenage girls cause their respective parents much concern when they start to become interested in boys. The parents' bickering about which girl is the worse influence causes more problems than it solves.", 
         "sentence_spans": [[0, 104], [104, 225], [225, 325]], 
         "sentence_labels": [0]}, 
        {"title": "Shirley Temple", "text": "Shirley Temple Black (April 23, 1928\u00a0\u2013 February 10, 2014) was an American actress, singer, dancer, businesswoman, and diplomat who was Hollywood's number one box-office draw as a child actress from 1935 to 1938. As an adult, she was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States.", "sentence_spans": [[0, 211], [211, 353]], "sentence_labels": [0, 1]}]
    """
    evidence_consolidated = []
    evidence_dict = {} 
    for i, e in enumerate(evidence_list):
        pid, sentidx = e
        title = pid_to_title[pid]
        if evidence_dict.get(title) is None:
            evidence_dict[title] = []
        evidence_dict[title].append(sentidx)
    for t in evidence_dict: 
        evidence_dict[t].sort()
        para = {}
        para['title'] = unescape(t)
        para['text'] = ''.join(corpus_dict[t]['sents'])
        para['sentence_spans'] = corpus_dict[t]['sentence_spans']
        para['sentence_labels'] = evidence_dict[t]
        evidence_consolidated.append(para)
    return evidence_consolidated


def create_samples(split, corpus_dict, pid_to_title, hover_seq, split_type='train'):
    """ Convert to standard input format, substituting Baleen pids with titles  
    split example:
    {'qid': 4,
     'question': 'Skagen Painter Peder Severin Krøyer favored naturalism along with Norderhov and the artist Ossian Elgström studied with in 1907.',
     'support_pids': [670219, 1513707, 811545],
     'support_facts': [[670219, 0], [670219, 1], [1513707, 1], [811545, 2]],
     'support_titles': ['Kristian Zahrtmann',
      'Peder Severin Krøyer',
      'Ossian Elgström'],
     'num_hops': 3,
     'label': 0,
     'uid': '824c8ffa-da36-45ea-9f68-0342e3893a63',
     'hpqa_id': '5ab7a86d5542995dae37e986'}    

    output:
    {"question": "[CLAIM] What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?", 
     "answers": ["Chief of Protocol"], [LABEL = SUPPORTED or NOT_SUPPORTED] 
     "src": 'hover',
     "type": "bridge",  [hover] 
     "pos_paras": [{"title": "Kiss and Tell (1945 film)", 
                    "text": "Kiss and Tell is a 1945 American comedy film starring then 17-year-old Shirley Temple as Corliss Archer. In the film, two teenage girls cause their respective parents much concern when they start to become interested in boys. The parents' bickering about which girl is the worse influence causes more problems than it solves.", 
                    "sentence_spans": [[0, 104], [104, 225], [225, 325]], 
                    "sentence_labels": [0]}, 
                   {"title": "Shirley Temple", "text": "Shirley Temple Black (April 23, 1928\u00a0\u2013 February 10, 2014) was an American actress, singer, dancer, businesswoman, and diplomat who was Hollywood's number one box-office draw as a child actress from 1935 to 1938. As an adult, she was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States.", "sentence_spans": [[0, 211], [211, 353]], "sentence_labels": [0, 1]}], 
     "neg_paras": [ [N/A] {"title": "A Kiss for Corliss", 
                    "text": "A Kiss for Corliss is a 1949 American comedy film directed by Richard Wallace and written by Howard Dimsdale. It stars Shirley Temple in her final starring role as well as her final film appearance. It is a sequel to the 1945 film \"Kiss and Tell\". \"A Kiss for Corliss\" was retitled \"Almost a Bride\" before release and this title appears in the title sequence. The film was released on November 25, 1949, by United Artists."}, 
                   {"title": "Kiss and Tell", "text": "Kiss and Tell or Kiss & Tell or Kiss n Tell may refer to:"}, {"title": "Secretary of State for Constitutional Affairs", "text": "The office of Secretary of State for Constitutional Affairs was a British Government position, created in 2003. Certain functions of the Lord Chancellor which related to the Lord Chancellor's Department were transferred to the Secretary of State. At a later date further functions were also transferred to the Secretary of State for Constitutional Affairs from the First Secretary of State, a position within the government held by the Deputy Prime Minister."}, {"title": "Joseph Kalite", "text": "Joseph Kalite (died 24 January 2014) was a Central African politician. As a government minister he either held the housing or health portfolio. Kalite, a Muslim, was reported to be killed by anti-balaka outside the Central Mosque in the capital Bangui during the Central African Republic conflict. He was killed with machetes on the day in Bangui after interim president Catherine Samba-Panza took power. At the time of the attack Kalite held no government position, nor did he under the S\u00e9l\u00e9ka rule. He was reported to have supported the rule of S\u00e9l\u00e9ka leader Michel Djotodia."}, {"title": "The Brave Archer and His Mate", "text": "The Brave Archer and His Mate, also known as The Brave Archer 4 and Mysterious Island, is a 1982 Hong Kong film adapted from Louis Cha's novels \"The Legend of the Condor Heroes\" and \"The Return of the Condor Heroes\". Together with \"Little Dragon Maiden\" (1983), \"The Brave Archer and His Mate\" is regarded as an unofficial sequel to the \"Brave Archer\" film trilogy (\"The Brave Archer\", \"The Brave Archer 2\" and \"The Brave Archer 3\")."}, {"title": "Little Dragon Maiden", "text": "Little Dragon Maiden, also known as The Brave Archer 5, is a 1983 Hong Kong film adapted from Louis Cha's novel \"The Return of the Condor Heroes\". \"Little Dragon Maiden\" and \"The Brave Archer and His Mate\" (1982) are seen as unofficial sequels to the \"Brave Archer\" film trilogy (\"The Brave Archer\", \"The Brave Archer 2\" and \"The Brave Archer 3\")."}, {"title": "The Brave Archer 3", "text": "The Brave Archer 3, also known as Blast of the Iron Palm, is a 1981 Hong Kong film adapted from Louis Cha's novel \"The Legend of the Condor Heroes\". The film was produced by the Shaw Brothers Studio and directed by Chang Cheh, starring Alexander Fu Sheng and Niu-niu in the lead roles. The film is the third part of a trilogy and was preceded by \"The Brave Archer\" (1977) and \"The Brave Archer 2\" (1978). The film has two unofficial sequels, \"The Brave Archer and His Mate\" (1982) and \"Little Dragon Maiden\" (1983), both of which were based on \"The Return of the Condor Heroes\". The theme song of the film, \"Say Cheung Kei\" (\u56db\u5f35\u6a5f), was composed by Chang Cheh, arranged by Joseph Koo and performed in Cantonese by Jenny Tseng."}, {"title": "The Brave Archer", "text": "The Brave Archer, also known as Kungfu Warlord, is a 1977 Hong Kong film adapted from Louis Cha's novel \"The Legend of the Condor Heroes\". The film was produced by the Shaw Brothers Studio and directed by Chang Cheh, starring Alexander Fu Sheng and Tanny Tien in the lead roles. The film is the first part of a trilogy and was followed by \"The Brave Archer 2\" (1978) and \"The Brave Archer 3\" (1981). The trilogy has two unofficial sequels, \"The Brave Archer and His Mate\" (1982) and \"Little Dragon Maiden\" (1983)."}, {"title": "Catherine Samba-Panza", "text": "Catherine Samba-Panza (born 26 June 1956) was interim President of the Central African Republic from 2014 to 2016. She was the first woman to hold the post of head of state in that country, as well as the eighth woman in Africa to do so. Prior to becoming head of state, she was Mayor of Bangui from 2013 to 2014."}, {"title": "The Brave Archer 2", "text": "The Brave Archer 2, also known as Kungfu Warlord 2, is a 1978 Hong Kong film adapted from Louis Cha's novel \"The Legend of the Condor Heroes\". The film was produced by the Shaw Brothers Studio and directed by Chang Cheh, starring Alexander Fu Sheng and Niu-niu in the lead roles. The film is the second part of a trilogy and was preceded by \"The Brave Archer\" (1977) and followed by \"The Brave Archer 3\" (1981). The trilogy has two unofficial sequels, \"The Brave Archer and His Mate\" (1982) and \"Little Dragon Maiden\" (1983)."}, {"title": "Charles Craft", "text": "Charles Craft (May 9, 1902 \u2013 September 19, 1968) was an English-born American film and television editor. Born in the county of Hampshire in England on May 9, 1902, Craft would enter the film industry in Hollywood in 1927. The first film he edited was the Universal Pictures silent film, \"Painting the Town\". Over the next 25 years, Craft would edit 90 feature-length films. In the early 1950s he would switch his focus to the small screen, his first show being \"Racket Squad\", from 1951\u201353, for which he was the main editor, editing 93 of the 98 episodes. He would work on several other series during the 1950s, including \"Meet Corliss Archer\" (1954), \"Science Fiction Theatre\" (1955\u201356), and \"Highway Patrol\" (1955\u201357). In the late 1950s and early 1960s he was one of the main editors on \"Sea Hunt\", starring Lloyd Bridges, editing over half of the episodes. His final film work would be editing \"Flipper's New Adventure\" (1964, the sequel to 1963's \"Flipper\". When the film was made into a television series, Craft would begin the editing duties on that show, editing the first 28 episodes before he retired in 1966. Craft died on September 19, 1968 in Los Angeles, California."}, {"title": "Lord High Treasurer", "text": "The post of Lord High Treasurer or Lord Treasurer was an English government position and has been a British government position since the Acts of Union of 1707. A holder of the post would be the third-highest-ranked Great Officer of State, below the Lord High Steward and the Lord High Chancellor."}, {"title": "Village accountant", "text": "The Village Accountant (variously known as \"Patwari\", \"Talati\", \"Patel\", \"Karnam\", \"Adhikari\", \"Shanbogaru\",\"Patnaik\" etc.) is an administrative government position found in rural parts of the Indian sub-continent. The office and the officeholder are called the \"patwari\" in Telangana, Bengal, North India and in Pakistan while in Sindh it is called \"tapedar\". The position is known as the \"karnam\" in Andhra Pradesh, \"patnaik\" in Orissa or \"adhikari\" in Tamil Nadu, while it is commonly known as the \"talati\" in Karnataka, Gujarat and Maharashtra. The position was known as the \"kulkarni\" in Northern Karnataka and Maharashtra. The position was known as the \"shanbogaru\" in South Karnataka."}, {"title": "Under-Secretary of State for War", "text": "The position of Under-Secretary of State for War was a British government position, first applied to Evan Nepean (appointed in 1794). In 1801 the offices for War and the Colonies were merged and the post became that of Under-Secretary of State for War and the Colonies. The position was re-instated in 1854 and remained until 1947, when it was combined with that of Financial Secretary to the War Office. In 1964 the War Office, Admiralty and Air Ministry were merged to form the Ministry of Defence, and the post was abolished."}, {"title": "Yeonguijeong", "text": "Yeonguijeong (] ) was a title created in 1400, during the Joseon Dynasty of Korea (1392-1910) and given to the Chief State Councillor as the highest government position of \"Uijeongbu\" (State Council). Existing for over 500 years, the function was handed over in 1895 during the Gabo Reform to the newly formed position of Prime Minister of Korea. Only one official at a time was appointed to the position and though was generally called \"Yeongsang\", was also referred to as \"Sangsang\", \"Sugyu\" or \"Wonbo\". Although, the title of Yeonguijeong was defined as the highest post in charge of every state affairs by law, its practical functions changed drastically depending on the particular King and whether that King's power was strong or weak."}, {"title": "Janet Waldo", "text": "Janet Marie Waldo (February 4, 1920 \u2013 June 12, 2016) was an American radio and voice actress. She is best known in animation for voicing Judy Jetson, Nancy in \"Shazzan\", Penelope Pitstop, and Josie in \"Josie and the Pussycats\", and on radio as the title character in \"Meet Corliss Archer\"."}, {"title": "Meet Corliss Archer (TV series)", "text": "Meet Corliss Archer is an American television sitcom that aired on CBS (July 13, 1951 - August 10, 1951) and in syndication via the Ziv Company from April to December 1954. The program was an adaptation of the radio series of the same name, which was based on a series of short stories by F. Hugh Herbert."}, {"title": "Centennial Exposition", "text": "The Centennial International Exhibition of 1876, the first official World's Fair in the United States, was held in Philadelphia, Pennsylvania, from May 10 to November 10, 1876, to celebrate the 100th anniversary of the signing of the Declaration of Independence in Philadelphia. Officially named the International Exhibition of Arts, Manufactures and Products of the Soil and Mine, it was held in Fairmount Park along the Schuylkill River on fairgrounds designed by Herman J. Schwarzmann. Nearly 10 million visitors attended the exhibition and thirty-seven countries participated in it."}, {"title": "Meet Corliss Archer", "text": "Meet Corliss Archer, a program from radio's Golden Age, ran from January 7, 1943 to September 30, 1956. Although it was CBS's answer to NBC's popular \"A Date with Judy\", it was also broadcast by NBC in 1948 as a summer replacement for \"The Bob Hope Show\". From October 3, 1952 to June 26, 1953, it aired on ABC, finally returning to CBS. Despite the program's long run, fewer than 24 episodes are known to exist."}], 
     "bridge": ["Shirley Temple"], [sequenced paras in nested list form - see calc_dependencies/recursive_replace for an example]
     "_id": "5a8c7595554299585d9e36b6"}  [hover uid]
    """
    out_samples = []
    for i, s in enumerate(split):
        sample = {'question': s['question'], 'answers': ['SUPPORTED'] if s['label']==1 else ['NOT_SUPPORTED'], 
                  'src': 'hover', 'type': 'multi', '_id': s['uid']}
        if split_type == 'train':
            seqs = calc_dependencies( hover_seq[ str(s["qid"]) ], verbose=False )     #remove_dups( hover_seq[ str(s["qid"]) ] ) # sometimes the same seq in different order appears, remove these
            sample['bridge'] = recursive_replace(seqs , pid_to_title)  # see recursive_replace docstring for example
        else: 
            sample['bridge'] = [unescape(st) for st in s['support_titles']] # no need to sequence dev..
        sample['num_hops'] = s['num_hops']  # note, numhops not present in other datasets, including in case it is useful downstream..
        sample['pos_paras'] = aggregate_evidence(s['support_facts'], corpus_dict, pid_to_title) 
        out_samples.append(sample)
    return out_samples


hover_dev_out = create_samples(hover_dev, baleen_dict, pid_to_title, hover_seq, split_type='dev')
hover_train_out = create_samples(hover_train, baleen_dict, pid_to_title, hover_seq, split_type='train')


# Add negative paras
docs = utils.load_jsonl(BQA_CORPUS)
titledict, dupdict = utils.build_title_idx(docs) # better to rebuild titledict as docs idxs changed after removal of docs with no paras.. 




def unescape_bridge(split):
    """ unescape the titles in bridge as neglected to do earlier..(and above code now fixed..)
    """
    for s in split:
        for paralist in s['bridge']:
            for i in range(len(paralist)):
                paralist[i] = unescape(paralist[i])
    return


def make_answer_list(split):
    """ turn answer into a list """
    for s in split:
        if type(s['answers']) != list:
            s['answers'] = [s['answers']]
    return


random.seed(42)
utils.add_neg_paras(docs, titledict, hover_dev_out)
utils.add_neg_paras(docs, titledict, hover_train_out)

unescape_bridge(hover_dev_out)
unescape_bridge(hover_train_out)

make_answer_list(hover_dev_out)
make_answer_list(hover_train_out)


utils.saveas_jsonl(hover_dev_out, UPDATED_DEV)
utils.saveas_jsonl(hover_train_out, UPDATED_TRAIN)

#hover_dev_out = utils.load_jsonl(UPDATED_DEV)
#hover_train_out = utils.load_jsonl(UPDATED_TRAIN)




#TODO Create qas file from dev
#{"question": "Were Scott Derrickson and Ed Wood of the same nationality?", 
#     "_id": "5a8b57f25542995d1e6f1371", 
#     "answer": ["yes"], 
#     "sp": ["Scott Derrickson", "Ed Wood"], 
#     "type": "comparison"}

def save_to_val_file(dev, outfile):
    print(f"saving to {outfile}..")
    out = []
    for sample in dev:
        sps = [p['title'] for p in sample['pos_paras']]

        sp_facts = []
        for para in sample['pos_paras']:
            for slabel in para['sentence_labels']:
                sp_facts.append( [para['title'], slabel] )  
        
        out_sample = {'question': sample['question'], '_id': sample['_id'], 'answer': sample['answers'], 
                      'sp': sps, 'sp_facts':sp_facts, 'type': sample['type'], 'src': sample['src']}
        out.append(out_sample)
    utils.saveas_jsonl(out, outfile)
    print('Finished save!')
    return

save_to_val_file(hover_dev_out, QAS_VAL_FILE_OUT)






