#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 13:46:27 2022

Convert PReasM (turning tables) datasets into standard uqa mtl format

@author: tim hartill



"""

import os
import json
import random
import smart_open  # import open
from tqdm import tqdm
import numpy as np
from html import unescape

import utils

OUT_DIR = '/data/thar011/data/unifiedqa/'


IN_DIR_BASE = '/home/thar011/data/turning_tables/reasoning_examples/'
IN_DEV = os.path.join(IN_DIR_BASE, 'dev')
IN_TRAIN = os.path.join(IN_DIR_BASE, 'train')
dev_files = os.listdir(IN_DEV)  # 15 files
dev_files = [os.path.join(IN_DEV, f) for f in dev_files]
train_files = os.listdir(IN_TRAIN)  # 15 files
train_files = [os.path.join(IN_TRAIN, f) for f in train_files]

for t, d in zip(train_files, dev_files):
    train = os.path.splitext(os.path.split(t)[1])[0]    
    dev = os.path.splitext(os.path.split(d)[1])[0] 
    print(f'Match:{train==dev}  Train:{train}  Dev:{dev}')  # all match



def read_jsonl_smart(infile):
    """ Read a compressed jsonl file
    """
    out_list = []
    print(f"Opening {infile}")
    with smart_open.open(infile, "r") as f:
        for l in tqdm(f):
            sample = json.loads(l)
            out_list.append(sample)
    print(f"{len(out_list)} samples")        
    return out_list


def compare_question_phrase(samples):
    ne_idxs = []
    for i, s in enumerate(samples):
        if s['question'] != s['phrase']:
            ne_idxs.append(i)
    if len(ne_idxs) != 0:
        print(f"Found {len(ne_idxs)} where Question != Phrase")
    return ne_idxs


def check_question_phrases(files, verbose=False):
    """ Check 'question' = 'phrase' key in all cases
    """
    ne_lists = []
    for f in files:
        samples = read_jsonl_smart(f)
        ne_idxs = compare_question_phrase(samples)
        ne_lists.append(ne_idxs)
        if verbose:
            print(f"QUESTION: {samples[0]['question']}")
            print(f"CONTEXT: {samples[0]['context']}")
            print(f"ANSWER: {samples[0]['answer']}")
    return ne_lists


ne_dev = check_question_phrases(dev_files, verbose=True)  # q = p for all
ne_train = check_question_phrases(train_files)  # q = p for all



def output_std_format(train_files, dev_files, out_dir_base):
    """ Output in standard q \\n context \\n tab answer  format
    """
    for t, d in zip(train_files, dev_files):
        train = os.path.splitext( os.path.split(t)[1] )[0]    
        dev = os.path.splitext( os.path.split(d)[1] )[0] 
        if train!=dev:
            print('Mismatch! Fix and rerun.')
            print(f'Match:{train==dev}  Train:{train}  Dev:{dev}')
            break
        samples_train = read_jsonl_smart(t)
        train_list = [utils.create_uqa_example(s['question'], s['context'], s['answer']) for s in samples_train]
        samples_dev = read_jsonl_smart(d)
        dev_list = [utils.create_uqa_example(s['question'], s['context'], s['answer']) for s in samples_dev]
        out_dir = os.path.join(out_dir_base, "tt_"+train)
        print(f"Creating {out_dir}")
        os.makedirs(out_dir, exist_ok=True)
        outfile = os.path.join(out_dir, 'train.tsv')
        print(f"Outputting train: {outfile}")
        with open(outfile, 'w') as f:
            f.write(''.join(train_list))
        outfile = os.path.join(out_dir, 'dev.tsv')
        print(f"Outputting dev: {outfile}")
        with open(outfile, 'w') as f:
            f.write(''.join(dev_list))
    print('Finished!')
    return

output_std_format(train_files, dev_files, OUT_DIR)



"""
ne_dev = check_question_phrases(dev_files, verbose=True)  # q = p for all
Opening /home/thar011/data/turning_tables/reasoning_examples/dev/temporal_comparison.gz
13323it [00:00, 30613.73it/s]
13323 samples
QUESTION: In Music videos of Ailee discography, what happened first: the Title was "Dont Touch Me" or the Title was "Mind Your Own Business"?
CONTEXT: In Music videos of Ailee discography: The title was "Is You" in 2018. The year when the title was "Ill Show You" was 2012. The year when the title was "Rewrite..If I Can" was 2018. The year when the title was "Mind Your Own Business" was 2015. The year when the title was "Dont Touch Me" was 2014. The year when the title was "Blue Spring" was 2018. The year when the title was "Reminiscing" was 2017. The year when the title was "Sweater" was 2019
ANSWER: the Title was "Dont Touch Me"
Opening /home/thar011/data/turning_tables/reasoning_examples/dev/only_quantifier.gz
14952it [00:00, 33573.83it/s]
14952 samples
QUESTION: In 2007 of Ebertfest, is Man of Flowers the only Title that has Starring Norman Kaye, Alyson Best, Chris Haywood?
CONTEXT: In 2007 of Ebertfest: The starring when the title was Searching for the Wrong-Eyed Jesus was Jim White, Harry Crews, Johnny Dowd. The starring when the title was Come Early Morning was Ashley Judd, Jeffrey Donovan. The starring when the title was Sadie Thompson was Gloria Swanson, Lionel Barrymore. The starring when the title was Gattaca was Ethan Hawke, Uma Thurman. The starring when the title was The Weather Man was Nicolas Cage, Hope Davis, Michael Caine. The starring when the title was Beyond the Valley of the Dolls was Dolly Read, Cynthia Myers, Marcia McBroom. The starring when the title was Stroszek was Bruno S., Eva Mattes, Clemens Scheitz. The starring when the title was Man of Flowers was Norman Kaye, Alyson Best, Chris Haywood. The starring when the title was La Dolce Vita was Marcello Mastroianni, Anita Ekberg, Anouk Aimée. The starring when the title was Moolaadé was Fatoumata Coulibaly, Maimouna Hélène Diarra, Salimata Traoré. The starring when the title was Freddie Mercury: the Untold Story was Freddie Mercury, Jer Bulsara, Kashmira Cooke. The starring when the title was Holes was Shia LaBeouf, Sigourney Weaver, Jon Voight. The starring when the title was Perfume: The Story of a Murderer was Ben Whishaw, Alan Rickman, Rachel Hurd-Wood, Dustin Hoffman
ANSWER: yes
Opening /home/thar011/data/turning_tables/reasoning_examples/dev/most_quantifier.gz
2758it [00:00, 34991.55it/s]
2758 samples
QUESTION: In 2010 of list of allure cover models, do most issue(s) have photographer michael thompson?
CONTEXT: In 2010 of List of Allure cover models: The photographer when the issue  was March was Norman Jean Roy. The photographer when the issue  was August was Michael Thompson. The photographer when the issue  was December was Michael Thompson. The photographer when the issue  was January was Michael Thompson. The photographer when the issue  was June was Greg Kadel. The photographer when the issue  was May was Michael Thompson. The photographer when the issue  was April was Carter Smith. The photographer when the issue  was July was Michael Thompson. The photographer when the issue  was September was Michael Thompson. The photographer when the issue  was October was Norman Jean Roy. The photographer when the issue  was February was Michael Thompson. The photographer when the issue  was November was Tom Munro
ANSWER: yes
Opening /home/thar011/data/turning_tables/reasoning_examples/dev/every_quantifier.gz
560it [00:00, 25715.59it/s]
560 samples
QUESTION: In records | manned records of scmaglev, does every date have type maglev?
CONTEXT: In Records | Manned records of SCMaglev: The type when the date was February 1995 was Maglev. The type when the date was 1972 was Maglev. The type when the date was 12 December 1997 was Maglev. The type when the date was November 1989 was Maglev. The type when the date was 14 April 1999 was Maglev. The type when the date was 21 April 2015 was Maglev. The type when the date was 16 April 2015 was Maglev. The type when the date was February 1987 was Maglev. The type when the date was 2 December 2003 was Maglev
ANSWER: yes
Opening /home/thar011/data/turning_tables/reasoning_examples/dev/numeric_comparison_boolean.gz
11663it [00:00, 38270.84it/s]
11663 samples
QUESTION: Was the pick # when the player was George Belotti lower than the pick # when the player was Bob Kilcullen in Round eight of 1957 NFL Draft?
CONTEXT: In Round eight of 1957 NFL Draft: The pick # when the player was George Belotti was 87. The pick # when the player was Ernie Pitts was 92. The pick # when the player was Jack Harmon was 90. The pick # when the player was Al Ward was 91. The pick # when the player was Bob Kilcullen was 96. The pick # when the player was Paul Lopata was 93. The pick # when the player was Charlie Bradshaw was 94. The pick # when the player was Don Gillis was 89
ANSWER: yes
Opening /home/thar011/data/turning_tables/reasoning_examples/dev/numeric_superlatives.gz
3540it [00:00, 22187.74it/s]
3540 samples
QUESTION: In Round eight of 1957 NFL Draft, which player has the highest pick #?
CONTEXT: In Round eight of 1957 NFL Draft: The pick # when the player was George Belotti was 87. The pick # when the player was Roy Hord Jr. was 88. The pick # when the player was Ernie Pitts was 92. The pick # when the player was Jack Harmon was 90. The pick # when the player was Johnny Bookman was 97. The pick # when the player was Charlie Bradshaw was 94. The pick # when the player was Al Ward was 91. The pick # when the player was Don Gillis was 89. The pick # when the player was Bob Kilcullen was 96. The pick # when the player was Paul Lopata was 93. The pick # when the player was Hal McElhaney was 86. The pick # when the player was Dave Liddick was 95
ANSWER: Johnny Bookman
Opening /home/thar011/data/turning_tables/reasoning_examples/dev/composition.gz
7632it [00:00, 26833.69it/s]
7632 samples
QUESTION: What was the Starring(s) when the Year was 2004 in 2007 of Ebertfest?
CONTEXT: In 2007 of Ebertfest: The starring when the title was Man of Flowers was Norman Kaye, Alyson Best, Chris Haywood. The title when the starring was Ashley Judd, Jeffrey Donovan was Come Early Morning. The title when the year was 2004 was Moolaadé. The year when the director was Tom Tykwer was 2006. The director when the title was Perfume: The Story of a Murderer was Tom Tykwer. The starring when the year was 1997 was Ethan Hawke, Uma Thurman. The starring when the title was Moolaadé was Fatoumata Coulibaly, Maimouna Hélène Diarra, Salimata Traoré. The director when the title was Gattaca was Andrew Niccol
ANSWER: Fatoumata Coulibaly, Maimouna Hélène Diarra, Salimata Traoré
Opening /home/thar011/data/turning_tables/reasoning_examples/dev/temporal_comparison_boolean.gz
13851it [00:00, 37572.06it/s]
13851 samples
QUESTION: In Music videos of Ailee discography, the Title was "Dont Touch Me" earlier than when the Title was "Mind Your Own Business"?
CONTEXT: In Music videos of Ailee discography: The year when the title was "Rewrite..If I Can" was 2018. The year when the title was "Sweater" was 2019. The year when the title was "Dont Touch Me" was 2014. The year when the title was "Mind Your Own Business" was 2015. The year when the title was "Ill Show You" was 2012. The title was "Is You" in 2018. The year when the title was "Blue Spring" was 2018. The year when the title was "Reminiscing" was 2017
ANSWER: yes
Opening /home/thar011/data/turning_tables/reasoning_examples/dev/temporal_superlatives.gz
2476it [00:00, 36900.65it/s]
2476 samples
QUESTION: In 2007 of Ebertfest, which title has the earliest year?
CONTEXT: In 2007 of Ebertfest: The title was Searching for the Wrong-Eyed Jesus in 2003. The title was Beyond the Valley of the Dolls in 1970. The title was Stroszek in 1977. The title was Gattaca in 1997. The year when the title was Holes was 2003. The title was Man of Flowers in 1983. The title was La Dolce Vita in 1960. The title was Freddie Mercury: the Untold Story in 2000. The year when the title was Moolaadé was 2004. The title was Sadie Thompson in 1928. The title was Come Early Morning in 2006. The title was Perfume: The Story of a Murderer in 2006. The title was The Weather Man in 2005
ANSWER: Sadie Thompson
Opening /home/thar011/data/turning_tables/reasoning_examples/dev/conjunction.gz
9846it [00:00, 28871.29it/s]
9846 samples
QUESTION: In Albums of Re:Stage!, what was the title when the artist was Ortensia and the note was TV animation insert song mini album?
CONTEXT: In Albums of Re:Stage!: The artist when the note was KiRaRe 1st album was KiRaRe. The title when the note was Stellamaris 1st album was Q.E.D.. The artist when the note was Tetrarkhia 1st mini album was Tetrarkhia. The titles when the artist was Ortensia were DRe:AMER -Ortensia ver. and Pullulate. The artist when the note was Ortensia 1st album was Ortensia. The notes when the artist was Stellamaris were Stellamaris 1st album and TV animation insert song mini album. The artist when the title was DRe:AMER -KiRaRe ver.- was KiRaRe. The titles when the note was TV animation insert song mini album were DRe:AMER -Stellamaris ver., DRe:AMER -Ortensia ver., and DRe:AMER -KiRaRe ver.-
ANSWER: DRe:AMER -Ortensia ver.
Opening /home/thar011/data/turning_tables/reasoning_examples/dev/counting.gz
13449it [00:00, 21725.17it/s]
13449 samples
QUESTION: In January of 1994 WTA Tour, how many champions have semifinalists Magdalena Maleeva  Wang Shi-ting?
CONTEXT: In January of 1994 WTA Tour: The semifinalists when the champions was  Ginger Helgeson 7–6 (7–4) , 6–3 was Julie Halard  Patricia Hy. The champions when the semifinalists was   Julie Halard  Patricia Hy were  Ginger Helgeson 7–6 (7–4) , 6–3 and  Patricia Hy  Mercedes Paz  6–4, 7–6 (7–4) . The semifinalists when the champions was  Kimiko Date 6–4, 6–2 was Patty Fendick  Gabriela Sabatini. The semifinalists when the champions was  Mana Endo 6–1, 6–7 (1–7) , 6–4 was Chanda Rubin  Kristie Boogert. The champions when the semifinalists was   Magdalena Maleeva  Wang Shi-ting were  Laura Golarsa  Natalia Medvedeva  6–3, 6–1 and  Lindsay Davenport 6–1, 2–6, 6–3
ANSWER: 2
Opening /home/thar011/data/turning_tables/reasoning_examples/dev/composition_2_hop.gz
10086it [00:00, 33492.51it/s]
10086 samples
QUESTION: What was the Starring(s) when the Title was The Weather Man in 2007 of Ebertfest?
CONTEXT: In 2007 of Ebertfest: The year when the director was Raoul Walsh was 1928. The director when the title was Holes was Andrew Davis. The director when the title was Man of Flowers was Paul Cox. The director when the title was The Weather Man was Gore Verbinski. The year when the director was Werner Herzog was 1977. The starring when the year was 1997 was Ethan Hawke, Uma Thurman. The year when the director was Paul Cox was 1983. The starring when the year was 2005 was Nicolas Cage, Hope Davis, Michael Caine. The year when the director was Gore Verbinski was 2005. The director when the title was Sadie Thompson was Raoul Walsh. The starring when the year was 1977 was Bruno S., Eva Mattes, Clemens Scheitz. The year when the director was Rudi Dolezal, Hannes Rossacher was 2000. The director when the title was Perfume: The Story of a Murderer was Tom Tykwer. The starring when the year was 1970 was Dolly Read, Cynthia Myers, Marcia McBroom. The starring when the year was 1928 was Gloria Swanson, Lionel Barrymore
ANSWER: Nicolas Cage, Hope Davis, Michael Caine
Opening /home/thar011/data/turning_tables/reasoning_examples/dev/arithmetic_superlatives.gz
5087it [00:00, 33221.89it/s]
5087 samples
QUESTION: What was the highest pick # when the position was Back in Round eight of 1957 NFL Draft?
CONTEXT: In Round eight of 1957 NFL Draft: The pick # when the player was Bob Kilcullen was 96. The pick # when the player was Al Ward was 91. The pick # when the player was Hal McElhaney was 86. The players when the position was Back were Hal McElhaney, Johnny Bookman, and Al Ward. The pick # when the player was Johnny Bookman was 97. The pick # when the player was Charlie Bradshaw was 94. The players when the position was End were Jack Harmon, Ernie Pitts, and Paul Lopata. The pick # when the player was Roy Hord Jr. was 88. The pick #s when the position was Center were 89 and 87. The players when the position was Tackle were Dave Liddick, Roy Hord Jr., Charlie Bradshaw, and Bob Kilcullen
ANSWER: 97
Opening /home/thar011/data/turning_tables/reasoning_examples/dev/arithmetic_addition.gz
2512it [00:00, 34915.58it/s]
2512 samples
QUESTION: What was the sum of the pick # when the nfl team was Chicago Bears in Round eight of 1957 NFL Draft?
CONTEXT: In Round eight of 1957 NFL Draft: The pick # when the nfl team was New York Giants was 97. The player when the nfl team was Green Bay Packers was George Belotti. The pick #s when the nfl team was Chicago Bears were 91 and 96. The players when the nfl team was Los Angeles Rams were Roy Hord Jr. and Charlie Bradshaw. The pick # when the player was Paul Lopata was 93. The pick # when the player was Dave Liddick was 95. The pick #s when the nfl team was Los Angeles Rams were 94 and 88
ANSWER: 187
Opening /home/thar011/data/turning_tables/reasoning_examples/dev/temporal_difference.gz
13301it [00:00, 24321.18it/s]13301 samples
QUESTION: In Music videos of Ailee discography, how much time had passed between when the Title was "Love Note" and when the Title was "Because Its Love"?
CONTEXT: In Music videos of Ailee discography: The title was "Ill Show You" in 2012. The title was "My Grown Up Christmas List" in 2012. The year when the title was "The Poem of Destiny" was 2019. The year when the title was "If You" was 2016. The title was "Room Shaker" in 2019. The year when the title was "Home" was 2016. The year when the title was "Because Its Love" was 2016. The year when the title was "Love Note" was 2012
ANSWER: 4 years
"""


