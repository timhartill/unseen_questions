#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 13:15:21 2023

@author: tim hartill

convert SCIFACT samples to std retriever training format.

{"question": "[CLAIM] What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?", 
 "answers": ["Chief of Protocol"], 
 "src": 'scifact',
 "type": "multi",  [fever] 
 "pos_paras": [{"title": "Kiss and Tell (1945 film)", 
                "text": "Kiss and Tell is a 1945 American comedy film starring then 17-year-old Shirley Temple as Corliss Archer. In the film, two teenage girls cause their respective parents much concern when they start to become interested in boys. The parents' bickering about which girl is the worse influence causes more problems than it solves.", 
                "sentence_spans": [[0, 104], [104, 225], [225, 325]], 
                "sentence_labels": [0]}, 
               {"title": "Shirley Temple", "text": "Shirley Temple Black (April 23, 1928\u00a0\u2013 February 10, 2014) was an American actress, singer, dancer, businesswoman, and diplomat who was Hollywood's number one box-office draw as a child actress from 1935 to 1938. As an adult, she was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States.", "sentence_spans": [[0, 211], [211, 353]], "sentence_labels": [0, 1]}], 
 "neg_paras": [ [N/A] {"title": "A Kiss for Corliss", 
                "text": "A Kiss for Corliss is a 1949 American comedy film directed by Richard Wallace and written by Howard Dimsdale. It stars Shirley Temple in her final starring role as well as her final film appearance. It is a sequel to the 1945 film \"Kiss and Tell\". \"A Kiss for Corliss\" was retitled \"Almost a Bride\" before release and this title appears in the title sequence. The film was released on November 25, 1949, by United Artists."}, 
               {"title": "Kiss and Tell", "text": "Kiss and Tell or Kiss & Tell or Kiss n Tell may refer to:"}, {"title": "Secretary of State for Constitutional Affairs", "text": "The office of Secretary of State for Constitutional Affairs was a British Government position, created in 2003. Certain functions of the Lord Chancellor which related to the Lord Chancellor's Department were transferred to the Secretary of State. At a later date further functions were also transferred to the Secretary of State for Constitutional Affairs from the First Secretary of State, a position within the government held by the Deputy Prime Minister."}, {"title": "Joseph Kalite", "text": "Joseph Kalite (died 24 January 2014) was a Central African politician. As a government minister he either held the housing or health portfolio. Kalite, a Muslim, was reported to be killed by anti-balaka outside the Central Mosque in the capital Bangui during the Central African Republic conflict. He was killed with machetes on the day in Bangui after interim president Catherine Samba-Panza took power. At the time of the attack Kalite held no government position, nor did he under the S\u00e9l\u00e9ka rule. He was reported to have supported the rule of S\u00e9l\u00e9ka leader Michel Djotodia."}, {"title": "The Brave Archer and His Mate", "text": "The Brave Archer and His Mate, also known as The Brave Archer 4 and Mysterious Island, is a 1982 Hong Kong film adapted from Louis Cha's novels \"The Legend of the Condor Heroes\" and \"The Return of the Condor Heroes\". Together with \"Little Dragon Maiden\" (1983), \"The Brave Archer and His Mate\" is regarded as an unofficial sequel to the \"Brave Archer\" film trilogy (\"The Brave Archer\", \"The Brave Archer 2\" and \"The Brave Archer 3\")."}, {"title": "Little Dragon Maiden", "text": "Little Dragon Maiden, also known as The Brave Archer 5, is a 1983 Hong Kong film adapted from Louis Cha's novel \"The Return of the Condor Heroes\". \"Little Dragon Maiden\" and \"The Brave Archer and His Mate\" (1982) are seen as unofficial sequels to the \"Brave Archer\" film trilogy (\"The Brave Archer\", \"The Brave Archer 2\" and \"The Brave Archer 3\")."}, {"title": "The Brave Archer 3", "text": "The Brave Archer 3, also known as Blast of the Iron Palm, is a 1981 Hong Kong film adapted from Louis Cha's novel \"The Legend of the Condor Heroes\". The film was produced by the Shaw Brothers Studio and directed by Chang Cheh, starring Alexander Fu Sheng and Niu-niu in the lead roles. The film is the third part of a trilogy and was preceded by \"The Brave Archer\" (1977) and \"The Brave Archer 2\" (1978). The film has two unofficial sequels, \"The Brave Archer and His Mate\" (1982) and \"Little Dragon Maiden\" (1983), both of which were based on \"The Return of the Condor Heroes\". The theme song of the film, \"Say Cheung Kei\" (\u56db\u5f35\u6a5f), was composed by Chang Cheh, arranged by Joseph Koo and performed in Cantonese by Jenny Tseng."}, {"title": "The Brave Archer", "text": "The Brave Archer, also known as Kungfu Warlord, is a 1977 Hong Kong film adapted from Louis Cha's novel \"The Legend of the Condor Heroes\". The film was produced by the Shaw Brothers Studio and directed by Chang Cheh, starring Alexander Fu Sheng and Tanny Tien in the lead roles. The film is the first part of a trilogy and was followed by \"The Brave Archer 2\" (1978) and \"The Brave Archer 3\" (1981). The trilogy has two unofficial sequels, \"The Brave Archer and His Mate\" (1982) and \"Little Dragon Maiden\" (1983)."}, {"title": "Catherine Samba-Panza", "text": "Catherine Samba-Panza (born 26 June 1956) was interim President of the Central African Republic from 2014 to 2016. She was the first woman to hold the post of head of state in that country, as well as the eighth woman in Africa to do so. Prior to becoming head of state, she was Mayor of Bangui from 2013 to 2014."}, {"title": "The Brave Archer 2", "text": "The Brave Archer 2, also known as Kungfu Warlord 2, is a 1978 Hong Kong film adapted from Louis Cha's novel \"The Legend of the Condor Heroes\". The film was produced by the Shaw Brothers Studio and directed by Chang Cheh, starring Alexander Fu Sheng and Niu-niu in the lead roles. The film is the second part of a trilogy and was preceded by \"The Brave Archer\" (1977) and followed by \"The Brave Archer 3\" (1981). The trilogy has two unofficial sequels, \"The Brave Archer and His Mate\" (1982) and \"Little Dragon Maiden\" (1983)."}, {"title": "Charles Craft", "text": "Charles Craft (May 9, 1902 \u2013 September 19, 1968) was an English-born American film and television editor. Born in the county of Hampshire in England on May 9, 1902, Craft would enter the film industry in Hollywood in 1927. The first film he edited was the Universal Pictures silent film, \"Painting the Town\". Over the next 25 years, Craft would edit 90 feature-length films. In the early 1950s he would switch his focus to the small screen, his first show being \"Racket Squad\", from 1951\u201353, for which he was the main editor, editing 93 of the 98 episodes. He would work on several other series during the 1950s, including \"Meet Corliss Archer\" (1954), \"Science Fiction Theatre\" (1955\u201356), and \"Highway Patrol\" (1955\u201357). In the late 1950s and early 1960s he was one of the main editors on \"Sea Hunt\", starring Lloyd Bridges, editing over half of the episodes. His final film work would be editing \"Flipper's New Adventure\" (1964, the sequel to 1963's \"Flipper\". When the film was made into a television series, Craft would begin the editing duties on that show, editing the first 28 episodes before he retired in 1966. Craft died on September 19, 1968 in Los Angeles, California."}, {"title": "Lord High Treasurer", "text": "The post of Lord High Treasurer or Lord Treasurer was an English government position and has been a British government position since the Acts of Union of 1707. A holder of the post would be the third-highest-ranked Great Officer of State, below the Lord High Steward and the Lord High Chancellor."}, {"title": "Village accountant", "text": "The Village Accountant (variously known as \"Patwari\", \"Talati\", \"Patel\", \"Karnam\", \"Adhikari\", \"Shanbogaru\",\"Patnaik\" etc.) is an administrative government position found in rural parts of the Indian sub-continent. The office and the officeholder are called the \"patwari\" in Telangana, Bengal, North India and in Pakistan while in Sindh it is called \"tapedar\". The position is known as the \"karnam\" in Andhra Pradesh, \"patnaik\" in Orissa or \"adhikari\" in Tamil Nadu, while it is commonly known as the \"talati\" in Karnataka, Gujarat and Maharashtra. The position was known as the \"kulkarni\" in Northern Karnataka and Maharashtra. The position was known as the \"shanbogaru\" in South Karnataka."}, {"title": "Under-Secretary of State for War", "text": "The position of Under-Secretary of State for War was a British government position, first applied to Evan Nepean (appointed in 1794). In 1801 the offices for War and the Colonies were merged and the post became that of Under-Secretary of State for War and the Colonies. The position was re-instated in 1854 and remained until 1947, when it was combined with that of Financial Secretary to the War Office. In 1964 the War Office, Admiralty and Air Ministry were merged to form the Ministry of Defence, and the post was abolished."}, {"title": "Yeonguijeong", "text": "Yeonguijeong (] ) was a title created in 1400, during the Joseon Dynasty of Korea (1392-1910) and given to the Chief State Councillor as the highest government position of \"Uijeongbu\" (State Council). Existing for over 500 years, the function was handed over in 1895 during the Gabo Reform to the newly formed position of Prime Minister of Korea. Only one official at a time was appointed to the position and though was generally called \"Yeongsang\", was also referred to as \"Sangsang\", \"Sugyu\" or \"Wonbo\". Although, the title of Yeonguijeong was defined as the highest post in charge of every state affairs by law, its practical functions changed drastically depending on the particular King and whether that King's power was strong or weak."}, {"title": "Janet Waldo", "text": "Janet Marie Waldo (February 4, 1920 \u2013 June 12, 2016) was an American radio and voice actress. She is best known in animation for voicing Judy Jetson, Nancy in \"Shazzan\", Penelope Pitstop, and Josie in \"Josie and the Pussycats\", and on radio as the title character in \"Meet Corliss Archer\"."}, {"title": "Meet Corliss Archer (TV series)", "text": "Meet Corliss Archer is an American television sitcom that aired on CBS (July 13, 1951 - August 10, 1951) and in syndication via the Ziv Company from April to December 1954. The program was an adaptation of the radio series of the same name, which was based on a series of short stories by F. Hugh Herbert."}, {"title": "Centennial Exposition", "text": "The Centennial International Exhibition of 1876, the first official World's Fair in the United States, was held in Philadelphia, Pennsylvania, from May 10 to November 10, 1876, to celebrate the 100th anniversary of the signing of the Declaration of Independence in Philadelphia. Officially named the International Exhibition of Arts, Manufactures and Products of the Soil and Mine, it was held in Fairmount Park along the Schuylkill River on fairgrounds designed by Herman J. Schwarzmann. Nearly 10 million visitors attended the exhibition and thirty-seven countries participated in it."}, {"title": "Meet Corliss Archer", "text": "Meet Corliss Archer, a program from radio's Golden Age, ran from January 7, 1943 to September 30, 1956. Although it was CBS's answer to NBC's popular \"A Date with Judy\", it was also broadcast by NBC in 1948 as a summer replacement for \"The Bob Hope Show\". From October 3, 1952 to June 26, 1953, it aired on ABC, finally returning to CBS. Despite the program's long run, fewer than 24 episodes are known to exist."}], 
 "bridge": [["Shirley Temple"]], [all paras]
 "_id": "5a8c7595554299585d9e36b6"}  [F + fever id]
}

"""

import os
import json
import random
from html import unescape
import copy
import random

import eval_metrics
import utils
from text_processing import normalize_unicode, convert_brc, replace_chars, create_sentence_spans, strip_accents

BASE_DIR = '/home/thar011/data/scifact/data/'
DEV = BASE_DIR+'claims_dev.jsonl'
TRAIN = BASE_DIR+'claims_train.jsonl'
SCIORIG_CORPUS = BASE_DIR+'corpus.jsonl'

UPDATED_DEV = BASE_DIR+'scifact_orig_dev_with_neg_and_sent_annots.jsonl'
UPDATED_TRAIN = BASE_DIR+'scifact_orig_train_with_neg_and_sent_annots.jsonl'

sciorig_corpus = utils.load_jsonl(SCIORIG_CORPUS)  # dict_keys(['doc_id', 'title', 'abstract', 'structured'])
for s in sciorig_corpus:
    s['abstract'] = [' '+t if i>0 else t for i, t in enumerate(s['abstract'])]
sci_abstracts_out = [{'title': v['title'], 'text': ''.join(v['abstract']), 
                      'sentence_spans': create_sentence_spans(v['abstract']), 'doc_id': v['doc_id']} for v in sciorig_corpus] # 5183
print(len(sci_abstracts_out))
SCI_ABSTRACTS_DICT = {t['doc_id']: t for t in sci_abstracts_out}


dev =  utils.load_jsonl(DEV)        # 300   dict_keys(['id', 'claim', 'evidence', 'cited_doc_ids'])
train =  utils.load_jsonl(TRAIN)    # 809


def process_data(split):
    """ Create retriever training dataset
    Input formats:
    {"id": 1099, "claim": "Statins decrease blood cholesterol.", "evidence": {}, "cited_doc_ids": [7662206]} NEI
    {"id": 1379, "claim": "Women with a higher birth weight are more likely to develop breast cancer later in life.", 
         "evidence": {"16322674": [{"sentences": [5], "label": "SUPPORT"}, {"sentences": [6], "label": "SUPPORT"}], 
                      "27123743": [{"sentences": [3], "label": "SUPPORT"}, {"sentences": [4], "label": "SUPPORT"}], 
                      "23557241": [{"sentences": [6], "label": "SUPPORT"}], 
                      "17450673": [{"sentences": [5], "label": "SUPPORT"}]}, 
         "cited_doc_ids": [16322674, 27123743, 23557241, 17450673]}
    
    Output format:
    {"question": "[CLAIM] What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?", 
     "answers": ["Chief of Protocol"], 
     "src": 'scifact',
     "type": "multi",  [fever] 
     "pos_paras": [{"title": "Kiss and Tell (1945 film)", 
                    "text": "Kiss and Tell is a 1945 American comedy film starring then 17-year-old Shirley Temple as Corliss Archer. In the film, two teenage girls cause their respective parents much concern when they start to become interested in boys. The parents' bickering about which girl is the worse influence causes more problems than it solves.", 
                    "sentence_spans": [[0, 104], [104, 225], [225, 325]], 
                    "sentence_labels": [0]}, 
                   {"title": "Shirley Temple", "text": "Shirley Temple Black (April 23, 1928\u00a0\u2013 February 10, 2014) was an American actress, singer, dancer, businesswoman, and diplomat who was Hollywood's number one box-office draw as a child actress from 1935 to 1938. As an adult, she was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States.", "sentence_spans": [[0, 211], [211, 353]], "sentence_labels": [0, 1]}], 
     "neg_paras": [ [N/A] {"title": "A Kiss for Corliss", 
                    "text": "A Kiss for Corliss is a 1949 American comedy film directed by Richard Wallace and written by Howard Dimsdale. It stars Shirley Temple in her final starring role as well as her final film appearance. It is a sequel to the 1945 film \"Kiss and Tell\". \"A Kiss for Corliss\" was retitled \"Almost a Bride\" before release and this title appears in the title sequence. The film was released on November 25, 1949, by United Artists."}, 
                   {"title": "Kiss and Tell", "text": "Kiss and Tell or Kiss & Tell or Kiss n Tell may refer to:"}, {"title": "Secretary of State for Constitutional Affairs", "text": "The office of Secretary of State for Constitutional Affairs was a British Government position, created in 2003. Certain functions of the Lord Chancellor which related to the Lord Chancellor's Department were transferred to the Secretary of State. At a later date further functions were also transferred to the Secretary of State for Constitutional Affairs from the First Secretary of State, a position within the government held by the Deputy Prime Minister."}, {"title": "Joseph Kalite", "text": "Joseph Kalite (died 24 January 2014) was a Central African politician. As a government minister he either held the housing or health portfolio. Kalite, a Muslim, was reported to be killed by anti-balaka outside the Central Mosque in the capital Bangui during the Central African Republic conflict. He was killed with machetes on the day in Bangui after interim president Catherine Samba-Panza took power. At the time of the attack Kalite held no government position, nor did he under the S\u00e9l\u00e9ka rule. He was reported to have supported the rule of S\u00e9l\u00e9ka leader Michel Djotodia."}, {"title": "The Brave Archer and His Mate", "text": "The Brave Archer and His Mate, also known as The Brave Archer 4 and Mysterious Island, is a 1982 Hong Kong film adapted from Louis Cha's novels \"The Legend of the Condor Heroes\" and \"The Return of the Condor Heroes\". Together with \"Little Dragon Maiden\" (1983), \"The Brave Archer and His Mate\" is regarded as an unofficial sequel to the \"Brave Archer\" film trilogy (\"The Brave Archer\", \"The Brave Archer 2\" and \"The Brave Archer 3\")."}, {"title": "Little Dragon Maiden", "text": "Little Dragon Maiden, also known as The Brave Archer 5, is a 1983 Hong Kong film adapted from Louis Cha's novel \"The Return of the Condor Heroes\". \"Little Dragon Maiden\" and \"The Brave Archer and His Mate\" (1982) are seen as unofficial sequels to the \"Brave Archer\" film trilogy (\"The Brave Archer\", \"The Brave Archer 2\" and \"The Brave Archer 3\")."}, {"title": "The Brave Archer 3", "text": "The Brave Archer 3, also known as Blast of the Iron Palm, is a 1981 Hong Kong film adapted from Louis Cha's novel \"The Legend of the Condor Heroes\". The film was produced by the Shaw Brothers Studio and directed by Chang Cheh, starring Alexander Fu Sheng and Niu-niu in the lead roles. The film is the third part of a trilogy and was preceded by \"The Brave Archer\" (1977) and \"The Brave Archer 2\" (1978). The film has two unofficial sequels, \"The Brave Archer and His Mate\" (1982) and \"Little Dragon Maiden\" (1983), both of which were based on \"The Return of the Condor Heroes\". The theme song of the film, \"Say Cheung Kei\" (\u56db\u5f35\u6a5f), was composed by Chang Cheh, arranged by Joseph Koo and performed in Cantonese by Jenny Tseng."}, {"title": "The Brave Archer", "text": "The Brave Archer, also known as Kungfu Warlord, is a 1977 Hong Kong film adapted from Louis Cha's novel \"The Legend of the Condor Heroes\". The film was produced by the Shaw Brothers Studio and directed by Chang Cheh, starring Alexander Fu Sheng and Tanny Tien in the lead roles. The film is the first part of a trilogy and was followed by \"The Brave Archer 2\" (1978) and \"The Brave Archer 3\" (1981). The trilogy has two unofficial sequels, \"The Brave Archer and His Mate\" (1982) and \"Little Dragon Maiden\" (1983)."}, {"title": "Catherine Samba-Panza", "text": "Catherine Samba-Panza (born 26 June 1956) was interim President of the Central African Republic from 2014 to 2016. She was the first woman to hold the post of head of state in that country, as well as the eighth woman in Africa to do so. Prior to becoming head of state, she was Mayor of Bangui from 2013 to 2014."}, {"title": "The Brave Archer 2", "text": "The Brave Archer 2, also known as Kungfu Warlord 2, is a 1978 Hong Kong film adapted from Louis Cha's novel \"The Legend of the Condor Heroes\". The film was produced by the Shaw Brothers Studio and directed by Chang Cheh, starring Alexander Fu Sheng and Niu-niu in the lead roles. The film is the second part of a trilogy and was preceded by \"The Brave Archer\" (1977) and followed by \"The Brave Archer 3\" (1981). The trilogy has two unofficial sequels, \"The Brave Archer and His Mate\" (1982) and \"Little Dragon Maiden\" (1983)."}, {"title": "Charles Craft", "text": "Charles Craft (May 9, 1902 \u2013 September 19, 1968) was an English-born American film and television editor. Born in the county of Hampshire in England on May 9, 1902, Craft would enter the film industry in Hollywood in 1927. The first film he edited was the Universal Pictures silent film, \"Painting the Town\". Over the next 25 years, Craft would edit 90 feature-length films. In the early 1950s he would switch his focus to the small screen, his first show being \"Racket Squad\", from 1951\u201353, for which he was the main editor, editing 93 of the 98 episodes. He would work on several other series during the 1950s, including \"Meet Corliss Archer\" (1954), \"Science Fiction Theatre\" (1955\u201356), and \"Highway Patrol\" (1955\u201357). In the late 1950s and early 1960s he was one of the main editors on \"Sea Hunt\", starring Lloyd Bridges, editing over half of the episodes. His final film work would be editing \"Flipper's New Adventure\" (1964, the sequel to 1963's \"Flipper\". When the film was made into a television series, Craft would begin the editing duties on that show, editing the first 28 episodes before he retired in 1966. Craft died on September 19, 1968 in Los Angeles, California."}, {"title": "Lord High Treasurer", "text": "The post of Lord High Treasurer or Lord Treasurer was an English government position and has been a British government position since the Acts of Union of 1707. A holder of the post would be the third-highest-ranked Great Officer of State, below the Lord High Steward and the Lord High Chancellor."}, {"title": "Village accountant", "text": "The Village Accountant (variously known as \"Patwari\", \"Talati\", \"Patel\", \"Karnam\", \"Adhikari\", \"Shanbogaru\",\"Patnaik\" etc.) is an administrative government position found in rural parts of the Indian sub-continent. The office and the officeholder are called the \"patwari\" in Telangana, Bengal, North India and in Pakistan while in Sindh it is called \"tapedar\". The position is known as the \"karnam\" in Andhra Pradesh, \"patnaik\" in Orissa or \"adhikari\" in Tamil Nadu, while it is commonly known as the \"talati\" in Karnataka, Gujarat and Maharashtra. The position was known as the \"kulkarni\" in Northern Karnataka and Maharashtra. The position was known as the \"shanbogaru\" in South Karnataka."}, {"title": "Under-Secretary of State for War", "text": "The position of Under-Secretary of State for War was a British government position, first applied to Evan Nepean (appointed in 1794). In 1801 the offices for War and the Colonies were merged and the post became that of Under-Secretary of State for War and the Colonies. The position was re-instated in 1854 and remained until 1947, when it was combined with that of Financial Secretary to the War Office. In 1964 the War Office, Admiralty and Air Ministry were merged to form the Ministry of Defence, and the post was abolished."}, {"title": "Yeonguijeong", "text": "Yeonguijeong (] ) was a title created in 1400, during the Joseon Dynasty of Korea (1392-1910) and given to the Chief State Councillor as the highest government position of \"Uijeongbu\" (State Council). Existing for over 500 years, the function was handed over in 1895 during the Gabo Reform to the newly formed position of Prime Minister of Korea. Only one official at a time was appointed to the position and though was generally called \"Yeongsang\", was also referred to as \"Sangsang\", \"Sugyu\" or \"Wonbo\". Although, the title of Yeonguijeong was defined as the highest post in charge of every state affairs by law, its practical functions changed drastically depending on the particular King and whether that King's power was strong or weak."}, {"title": "Janet Waldo", "text": "Janet Marie Waldo (February 4, 1920 \u2013 June 12, 2016) was an American radio and voice actress. She is best known in animation for voicing Judy Jetson, Nancy in \"Shazzan\", Penelope Pitstop, and Josie in \"Josie and the Pussycats\", and on radio as the title character in \"Meet Corliss Archer\"."}, {"title": "Meet Corliss Archer (TV series)", "text": "Meet Corliss Archer is an American television sitcom that aired on CBS (July 13, 1951 - August 10, 1951) and in syndication via the Ziv Company from April to December 1954. The program was an adaptation of the radio series of the same name, which was based on a series of short stories by F. Hugh Herbert."}, {"title": "Centennial Exposition", "text": "The Centennial International Exhibition of 1876, the first official World's Fair in the United States, was held in Philadelphia, Pennsylvania, from May 10 to November 10, 1876, to celebrate the 100th anniversary of the signing of the Declaration of Independence in Philadelphia. Officially named the International Exhibition of Arts, Manufactures and Products of the Soil and Mine, it was held in Fairmount Park along the Schuylkill River on fairgrounds designed by Herman J. Schwarzmann. Nearly 10 million visitors attended the exhibition and thirty-seven countries participated in it."}, {"title": "Meet Corliss Archer", "text": "Meet Corliss Archer, a program from radio's Golden Age, ran from January 7, 1943 to September 30, 1956. Although it was CBS's answer to NBC's popular \"A Date with Judy\", it was also broadcast by NBC in 1948 as a summer replacement for \"The Bob Hope Show\". From October 3, 1952 to June 26, 1953, it aired on ABC, finally returning to CBS. Despite the program's long run, fewer than 24 episodes are known to exist."}], 
     "bridge": [["Shirley Temple"]], [all paras]
     "_id": "5a8c7595554299585d9e36b6"}  [F + fever id]
    }    
    
    
    if cited_doc_id not in evidence -> add this doc to neg paras
    Add pos doc minus all gold sents -> neg paras
    Add random negs not in cited_doc_ids
    """
    outlist = []
    for i, s in enumerate(split):
        ev = s.get('evidence')
        if ev is not None and ev != {}:  # skip NEIs
            doc_id_set = set()
            curr_out_list = []
            neg_paras = []  # note we can use the same neg paras for each outputted retriever sample for a given scifact sample
            for doc_id in ev:
                doc_id_set.add(int(doc_id))                
                #label_set = set()
                sentence_labels = []
                for sent in ev[doc_id]:
                    if sent['label'].strip() == 'SUPPORT':  #take label as last one in ev[doc_id] list since same label for all sents per doc in train/dev
                        ans = 'yes'
                    else:
                        ans = 'no'
                    sentence_labels += sent['sentences']  # technically each item is an evidence set so could have multiple evidence sets per doc but here we simplify by aggregating all gold sents as 1 evidence set (see https://github.com/allenai/scifact/blob/master/doc/evaluation.md)
                sentence_labels.sort()
                para = copy.deepcopy(SCI_ABSTRACTS_DICT[int(doc_id)])
                para['sentence_labels'] = sentence_labels
                neg_txt = [para['text'][s:e] for i,(s,e) in enumerate(para['sentence_spans']) if i not in sentence_labels]   # strip gold sents
                neg_txt = ''.join(neg_txt).strip()
                neg_paras.append( {'title': para['title'], 'text': neg_txt, 'src': 'nogold'} )
                out = {'question': s['claim'], 'answers':[ans], 'src': 'scifact', 'type': 'multi', '_id': str(s['id'])+'_'+str(doc_id),
                       'pos_paras': [para],
                       'neg_paras': [],
                       'bridge': [[para['title']]]}
                curr_out_list.append(out)
            for doc_id in s['cited_doc_ids']:  # add docids cited but deemed insufficient ev as adv negs
                if int(doc_id) not in doc_id_set:
                    neg = SCI_ABSTRACTS_DICT[int(doc_id)]
                    neg_paras.append( {'title': neg['title'], 'text': neg['text'], 'src':'adv'} )
                    doc_id_set.add(int(doc_id))
            if neg_paras != []:
                for out in curr_out_list:
                    out['neg_paras'] += neg_paras
              
            all_ids = list(SCI_ABSTRACTS_DICT.keys())
            for out in curr_out_list:  # finally, top up negatives with random negs if < 10
                while len(out['neg_paras']) < 10:
                    rand_id = random.choice(all_ids)
                    if rand_id not in doc_id_set:
                        neg = SCI_ABSTRACTS_DICT[rand_id]
                        out['neg_paras'].append({'title': neg['title'], 'text': neg['text'], 'src':'rd'})
            outlist += curr_out_list
    return outlist                    
                    
                
random.seed(42)      
outlist_dev = process_data(dev)     #209
outlist_train = process_data(train) #564

utils.saveas_jsonl(outlist_dev, UPDATED_DEV)
utils.saveas_jsonl(outlist_train, UPDATED_TRAIN)


