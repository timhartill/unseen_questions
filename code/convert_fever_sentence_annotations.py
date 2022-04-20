#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:27:21 2022

@author: tim hartill

Combine FEVER corpus sentence breakdown and FEVER train/dev sentence-level annotations 
and output in same format as "BQA" HPQA sentence level annotations

Download fever corpus and train/dev files from https://fever.ai/dataset/fever.html

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

NOTES:
    - In FEVER, correctly recalling ANY evidence set counts as a win but in HPQ recalling both paras of a single evidence set is the win
        - hence we make 1 sample out of each unique evidence set: q1 set1, q1 set2
        - where multiple sentences in a single doc are pointed to in difft evidence sets we label all such sentences in the doc as positive and consolidate
    - Since we are building this dataset to train a sentence prediction model we skip "NOT SUPPORTED" claims.
    - We retain REFUTED claims in the belief that we are marking sentences as evidential toward deriving an answer rather than just positive examples ie "positive" means "evidential" not necessarily "supportive"
"""
import os
import json
from html import unescape

import utils
from text_processing import normalize_unicode, convert_brc, replace_chars, create_sentence_spans


DEV = '/home/thar011/data/fever/shared_task_dev.jsonl'
TRAIN = '/home/thar011/data/fever/train.jsonl'
PROCESSED_CORPUS_DIR = '/home/thar011/data/fever/wiki-pages'

UPDATED_DEV = '/home/thar011/data/fever/fever_dev_with_sent_annots.jsonl'
UPDATED_TRAIN = '/home/thar011/data/fever/fever_train_with_sent_annots.jsonl'

fever_dev = utils.load_jsonl(DEV)      # 19998 dict_keys(['id', 'verifiable', 'label', 'claim', 'evidence'])
fever_train =  utils.load_jsonl(TRAIN) # 145449 dict_keys(['id', 'verifiable', 'label', 'claim', 'evidence'])


def convert_text_and_lines(sample):
    """ Standardize FEVER text, calculate sentence spans.
    """   
    new_sample = {}
    new_sample['title'] = sample['id']
    lines = normalize_unicode(convert_brc(sample['lines'])).split('\n')
    newlines = []
    for line in lines:
        line_split = line.split('\t')
        if len(line_split) > 1:
            newline = line_split[1]
            if newline != '':
                newline = newline.strip()
                if newline[-1] not in ['.', '!', '?']:
                    newline += '.'
                newlines.append(' ' + replace_chars(newline))
    if newlines != []:
        newlines[0] = newlines[0][1:]  # follow hpqa form where 1st sent doesn't start with space and all subsequent ones do
    new_sample['text'] = ''.join(newlines)
    new_sample['sentence_spans'] = create_sentence_spans(newlines)    
    return new_sample 
    

# build consolidated corpus
wiki_jsonl_files = os.listdir(PROCESSED_CORPUS_DIR)  # 109 files
wiki_fever = []
for i,f in enumerate(wiki_jsonl_files):
    curr_wiki = utils.load_jsonl(os.path.join(PROCESSED_CORPUS_DIR, f))
    wiki_fever.extend(curr_wiki)
print(f"Combined abstracts: {len(wiki_fever)}")  # Combined abstracts: 5416537  dict_keys(['id', 'text', 'lines'])

wiki_dict = {}
for i, sample in enumerate(wiki_fever):
    new_sample = convert_text_and_lines(sample)    
    wiki_dict[new_sample['title']] = new_sample
    if i % 50000 == 0:
        print(f'Processed: {i}')
print(f"Num docs: {len(wiki_dict)}")
        

#TODO output multiple samples one per consolidated evidence set
#TODO consolidate all single evidences into separate samples 
#eg [ [titleA, 0], [titleA, 5], [titleB, 6] ] -> {q, [titleA:[0,5]]} & {q, [titleb:[6]]}
#TODO consolidate all multi evidense separately: eg [ [[A,4], [B,0], [B,2] ] ] -> {q, [A:[4], B:[0,2]]}

def get_title_sidx(entry):
    """ eg [131371, 146150, 'Telemundo', 4]
    """
    return entry[2], entry[3]


def process_evidence(evidence_list):
    """ process evidence list [ [[single ev1]], [[double ev1], [double ev2]], ..   ]
    into eg: [ {singletitle:[0]}, {'doublet1': [2], 'doublet2': [4,5]} ]
    """
    evidence_consolidated = []
    evidence_single = {}
    for i, e in enumerate(evidence_list):
        if len(e) == 1:
            title, sentidx = get_title_sidx(e[0])
            if evidence_single.get(title) is None:
                evidence_single[title] = []
            if sentidx not in evidence_single[title]:
                evidence_single[title].append(sentidx)
        elif len(e) > 1:
            evidence_multi = {}
            for ev in e:
                title, sentidx = get_title_sidx(ev)
                if evidence_multi.get(title) is None:
                    evidence_multi[title] = []
                if sentidx not in evidence_multi[title]:
                    evidence_multi[title].append(sentidx)  # Note: technically correct = predict all separate sentences in same para but for our sentence pred model this is ok
            if len(evidence_multi) == 1: # we have multiple sentences from one doc and need to predict all of them to be correct..
                t = list(evidence_multi.keys())[0]
                ev_multi_split = {}
                for j, sentidx in enumerate(evidence_multi[t]):
                    ev_multi_split[t+'_vvv'+str(j)] = [sentidx]
                evidence_multi = ev_multi_split                          
            for t in evidence_multi:
                evidence_multi[t].sort()            
            evidence_consolidated.append(evidence_multi) # claim correct if retrieved evidence matchs all paras (but any single sentence in each para?) 
    if evidence_single != {}:
        for t in evidence_single: # claim correct if retrieved evidence matches any para (and any single sentence in para)
            evidence_single[t].sort()
            evidence_consolidated.append( {t: evidence_single[t]} )  
    return evidence_consolidated


            

def create_samples(split, wiki_dict):
    """ Create samples from split
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
    
    NOTES:
        - In FEVER, correctly recalling ANY evidence set counts as a win but in HPQ recalling both paras of a single evidence set is the win
            - hence we make 1 sample out of each unique evidence set: q1 set1, q1 set2
            - where multiple sentences in a single doc are pointed to in difft evidence sets we label all such sentences in the doc as positive and consolidate
        - Since we are building this dataset to train a sentence prediction model we skip "NOT SUPPORTED" claims.
        - We retain REFUTED claims in the belief that we are marking sentences as evidential toward deriving an answer rather than just positive examples ie "positive" means "evidential" not necessarily "supportive"
    
    """
    out_samples = []
    missing_titles = []
    for i,s in enumerate(split):
        label = s['label']
        if label != 'NOT ENOUGH INFO':
            ans = [label]
            q = normalize_unicode(convert_brc( s['claim'] ) )
            fid = str(s['id'])
            evidence_consolidated = process_evidence(s['evidence'])  # eg [{'Telemundo': [4], 'Hispanic_and_Latino_Americans': [0]}, {'Telemundo': [0, 1, 5]}]
            pos_paras_list = []
            for ev in evidence_consolidated:  # add text and sentence span annots
                pos_paras_single_set = []
                for orig_title in ev:
                    title = orig_title
                    chk = title.find('_vvv')  # kludge for case where multievidence from single title
                    if chk > -1:
                        title = title[:chk]    
                    new_title = normalize_unicode(convert_brc(unescape(title))).replace('_', ' ')
                    w = wiki_dict.get(title)
                    if w is not None:
                        pos_para_dict = {}
                        pos_para_dict['title'] = new_title
                        pos_para_dict['text'] = w['text']
                        pos_para_dict['sentence_spans'] = w['sentence_spans']
                        pos_para_dict['sentence_labels'] = ev[orig_title]
                        pos_paras_single_set.append(pos_para_dict)
                    else:
                        missing_titles.append({'title': title, 'split_idx':i})
                pos_paras_list.append(pos_paras_single_set)
            for j, pos_paras in enumerate(pos_paras_list):
                sample = {'question': q, 'answers': ans, 'src': 'fever', 'type': 'fever', '_id': fid+'_'+str(i)}
                sample['bridge'] = [t['title'] for t in pos_paras]
                sample['pos_paras'] = pos_paras
                out_samples.append(sample)
        if i % 25000 == 0:
            print(f"Processed: {i}")
    return out_samples, missing_titles
                    
                        
fever_dev_out, dev_missing = create_samples(fever_dev, wiki_dict)         # 173 missing       
fever_train_out, train_missing = create_samples(fever_train, wiki_dict)   # 1283 missing             
     

fever_dev[19799]
{'id': 108345,
 'verifiable': 'VERIFIABLE',
 'label': 'REFUTES',
 'claim': 'Raven-Symoné refuses to be an actress.',
 'evidence': [[[127152, 141660, 'Raven-Symoné', 0]],
  [[127152, 141661, 'Raven-Symoné', 1]],
  [[127152, 141662, 'Raven-Symoné', 6]],
  [[127152, 141663, 'Raven-Symoné', 7]],
  [[127152, 141664, 'Raven-Symoné', 8],
   [127152, 141664, 'This_Is_My_Time_-LRB-Raven-Symoné_album-RRB-', 0]],
  [[127152, 141665, 'Raven-Symoné', 9],
   [127152, 141665, 'Raven-Symoné_-LRB-album-RRB-', 0]],
  [[127152, 141666, 'Raven-Symoné', 17]],
  [[127152, 141667, 'Raven-Symoné', 18]],
  [[127152, 141668, 'Raven-Symoné', 19]],
  [[127152, 141669, 'Raven-Symoné', 23]],
  [[127152, 141670, 'Raven-Symoné', 24]]]}
       
wlist = [wiki_dict[w] for w in wiki_dict if 'Raven-Symon' in wiki_dict[w]['title']]  #7 accent in dev/train is wrong 
    
    
{'title': 'This_Is_My_Time_-LRB-Raven-Symoné_album-RRB-',
 'text': "This Is My Time is the third studio album by American singer-songwriter and actress Raven-Symoné, released in the United States on September 21, 2004 by Hollywood Records. The album is her debut with the Disney-owned label and debuted at number fifty-one on the U.S. Billboard 200, with 19,000 copies sold in its first week. It nevertheless became Raven-Symoné 's best-selling solo album, selling 235,000 copies up to February 2, 2007 in the U.S., according to Nielsen SoundScan.",
 'sentence_spans': [[0, 172], [172, 325], [325, 481]]}

{'title': 'Raven-Symoné',
 'text': "Raven-Symoné Christina Pearman ( [ ˈreɪ.vən _ sɪˈmoʊn ] born December 10, 1985 ), sometimes credited as Raven, is an American actress, singer, songwriter, television personality, and producer. She first appeared on television in 1989 on The Cosby Show as Olivia Kendall. She released her debut album, Here 's to New Dreams in 1993; the single, `` That 's What Little Girls Are Made Of '' charted number 68 on the US Billboard Hot 100. The next album, Undeniable, was released on May 4, 1999. Raven-Symoné appeared in several successful television series, such as The Cosby Show and Hangin ' with Mr. Cooper, in the late 1980s and early 1990s. From 2003 to 2007, she starred in the Disney Channel series, That 's So Raven in which she played Raven Baxter, a psychic teenager who tried her best to keep her psychic powers a secret. During her time on That 's So Raven, Raven-Symoné released her third studio album, This is My Time ( 2004 ) which was her best selling solo album to date, charting at number 51 on the Billboard 200. A year after the end of That 's So Raven, she released her fourth studio album, Raven-Symoné ( 2008 ). The album peaked at number 159 on the Billboard 200. During 2003 to 2006, she participated in four soundtracks from Disney, RIAA-certified double-platinum album, The Cheetah Girls ( 2003 ), RIAA-certified gold album, That 's So Raven ( 2004 ), That 's So Raven Too! ( 2006 ) and RIAA-certified platinum album, The Cheetah Girls 2 ( 2006 ). The soundtracks sold a combined 4.1 million copies in the U.S. alone. As of April 2008, Raven-Symoné has sold 314,000 albums in the United States. Raven-Symoné transitioned to a film career, starring in several films aimed at young audiences, including Dr. Dolittle ( 1998 ), Dr. Dolittle 2 ( 2001 ), College Road Trip ( 2008 ), and successful television films, including Zenon: Girl of the 21st Century ( 1999 ), The Cheetah Girls ( 2003 ), its sequel Cheetah Girls 2 ( 2006 ), For One Night ( 2006 ), Revenge of the Bridesmaids ( 2010 ). Raven has also lent her voice to the animated series Kim Possible, for the character Monique and films such as Disney 's Tinker Bell. In 2011, Symoné starred in the short-lived ABC Family comedy series State of Georgia as Georgia Chamberlain, an aspiring actress with a huge ego who moves to New York City to try her hand at an acting career. In 2012, Raven-Symoné ranked number nine on VH1 `` 100 Greatest Kid Stars Of All Time '' list. In 2015, Raven-Symoné joined the cast of the new hit primetime drama Empire in the recurring role of Olivia, the vengeful ex-wife of one of the main characters. She stars alongside Terrence Howard, Malik Yoba, and Taraji P. Henson. In June 2015, she joined the ABC panel show The View, after a period of guest-hosting.",
 'sentence_spans': [[0, 193],
  [193, 271],
  [271, 435],
  [435, 492],
  [492, 644],
  [644, 831],
  [831, 1031],
  [1031, 1135],
  [1135, 1188],
  [1188, 1401],
  [1401, 1475],
  [1475, 1545],
  [1545, 1623],
  [1623, 2017],
  [2017, 2151],
  [2151, 2361],
  [2361, 2457],
  [2457, 2619],
  [2619, 2690],
  [2690, 2777]]}

