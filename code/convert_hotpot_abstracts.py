#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 18:35:27 2022

@author: tim hartill

MDR: Convert HotpotQA abstracts into a jsonl file from the MDR output file to the input format for encoding

output:
[ {'title': 'the title', 'text': 'the text'} ]

eg:
[{'title': 'Anarchism',
   'text': 'Anarchism is a political philosophy that advocates self-governed societies based on voluntary institutions. These are often described as stateless societies, although several authors have defined them more specifically as institutions based on non-hierarchical free associations. Anarchism holds the state to be undesirable, unnecessary and harmful.'}]

From Hotpot raw:  sentence offsets: 1st sentence no trailing space then all other sentences begin with " "
[{'id': '12',
  'url': 'https://en.wikipedia.org/wiki?curid=12',
  'title': 'Anarchism',
  'text': ['Anarchism is a political philosophy that advocates self-governed societies based on voluntary institutions.',
   ' These are often described as stateless societies, although several authors have defined them more specifically as institutions based on non-hierarchical free associations.',
   ' Anarchism holds the state to be undesirable, unnecessary and harmful.'],
  'charoffset': [[[0, 9], [10, 12], [13, 14], ...]],...]
  'text_with_links': ['Anarchism is a <a href="political%20philosophy">political philosophy</a> that advocates <a href="self-governance">self-governed</a> societies based on voluntary institutions.',
   ' These are often described as <a href="stateless%20society">stateless societies</a>, although several authors have defined them more specifically as institutions based on non-<a href="Hierarchy">hierarchical</a> <a href="Free%20association%20%28communism%20and%20anarchism%29">free associations</a>.',
   ' Anarchism holds the <a href="state%20%28polity%29">state</a> to be undesirable, unnecessary and harmful.'],
  'charoffset_with_links': [[[0, 9], ...]   

AISO: Fool around with the original hpqa multiple bz2 file format

output: .tsv file with header: id	text	title   hyperlinks	sentence_spans
eg:
NOTE: Longer than HPQA version!! 
anchor text = text in text with key = href with percent encoding
12_0	
Anarchism is a political philosophy that advocates self-governed societies based on voluntary institutions. These are often described as stateless societies, although several authors have defined them more specifically as institutions based on non-hierarchical free associations. Anarchism holds the state to be undesirable, unnecessary and harmful. While anti-statism is central, anarchism specifically entails opposing authority or hierarchical organisation in the conduct of all human relations, including—but not limited to—the state system. Anarchism is usually considered a far-left ideology and much of anarchist economics and anarchist legal philosophy reflects anti-authoritarian interpretations of communism, collectivism, syndicalism, mutualism or participatory economics.	
Anarchism	
{"Hierarchy": [{"anchor_text": "hierarchical", "span": [248, 260]}], "Free association (communism and anarchism)": [{"anchor_text": "free associations", "span": [261, 278]}], "Far-left politics": [{"anchor_text": "far-left", "span": [580, 588]}], "Libertarian socialism": [{"anchor_text": "anti-authoritarian interpretations", "span": [670, 704]}], "Mutualism (economic theory)": [{"anchor_text": "mutualism", "span": [746, 755]}]}	
[[0, 108], [108, 279], [279, 349]]    # sentence spans are always contiguous but " " can be at beginning or end
    
    
and: 
    
93129_0	
Crawford County is a county located in the U.S. state of Ohio. As of the 2010 census, the population was 43,784. The approximate population as of 2014 is 42,480, causing a -3.00% change over the past 4 years, according to the United States Census Bureau. Its county seat is Bucyrus. The county was created in 1820 and later organized in 1836. It was named for Colonel William Crawford, a soldier during the American Revolution. Crawford County comprises the Bucyrus, OH Micropolitan Statistical Area, which is also included in the Mansfield-Ashland-Bucyrus, OH Combined Statistical Area. According to the U.S. Census Bureau, the county has a total area of 403 sqmi , of which 402 sqmi is land and 0.9 sqmi (0.2%) is water. It is the fourth-smallest county in Ohio by total area.	
Crawford County, Ohio	
{"County (United States)": [{"anchor_text": "county", "span": [21, 27]}], "U.S. state": [{"anchor_text": "U.S. state", "span": [43, 53]}], "Ohio": [{"anchor_text": "Ohio", "span": [57, 61]}], "2010 United States Census": [{"anchor_text": "2010 census", "span": [73, 84]}], "Bucyrus, Ohio": [{"anchor_text": "Bucyrus", "span": [274, 281]}], "William Crawford (soldier)": [{"anchor_text": "William Crawford", "span": [368, 384]}], "Mansfield, Ohio": [{"anchor_text": "Mansfield", "span": [531, 540]}], "Ashland, Ohio": [{"anchor_text": "Ashland", "span": [541, 548]}], "Mansfield Metropolitan Statistical Area": [{"anchor_text": "Combined Statistical Area", "span": [561, 586]}]}	
[[0, 63], [63, 113], [113, 254], [254, 282], [282, 342], [342, 427]]

"""

import os
import utils
import json
import bz2
import glob
from urllib.parse import unquote  #convert percent encoding eg %28%20%29 -> ( )   quote does opposite
from html import unescape
import copy


####### MDR:

OUTDIR = '/data/thar011/gitrepos/compgen_mdr/data/hpqa_raw_tim'


mdr_hpqa = utils.loadas_json('/data/thar011/gitrepos/compgen_mdr/data/hotpot_index/wiki_id2doc.json') # 5233329

mdr_out = [{'title': v['title'], 'text': v['text']} for v in mdr_hpqa.values() if v['text'].strip() != ''] # 5233235 strips blanks

utils.saveas_jsonl(mdr_out, os.path.join(OUTDIR, 'hpqa_abstracts_tim.jsonl'))


#test
#datatest = [json.loads(l) for l in open(os.path.join(OUTDIR, 'hpqa_abstracts_tim.jsonl')).readlines()]


###### AISO
INDIR_BASE = '/data/thar011/gitrepos/compgen_mdr/data/hpqa_raw_tim/enwiki-20171001-pages-meta-current-withlinks-abstracts'
AISO_FILE = '/data/thar011/gitrepos/AISO/data/corpus/hotpot-paragraph.strict.tjh.tsv'


#tstfile = '/AA/wiki_00.bz2'

def load_bz2_to_jsonl(infile, verbose=False):
    source_file = bz2.BZ2File(infile, "r")
    out = []
    count = 0
    for line in source_file:
        count += 1
        out.append(json.loads(line))
    source_file.close()
    if verbose:
        print(f"Read {count} rows from {infile}")
    return out


def get_para(paras, idx=None, title=None):
    if idx is not None:
        out = [p for p in paras if p['id'] == idx]
    elif title is not None:
        out = [p for p in paras if p['title'] == title]
    return out[0]    

def adj_token_boundaries(toks):
    """ adjust token boundaries s.t. they are contiguous
    toks = flattened list of alternating start/stop tokens
    """
    numtoks = len(toks)
    if numtoks > 0:
        toks[0] = 0  
    for i in range(0, numtoks, 2):  #i=start, i+1=end
        if i < numtoks-2:
            toks[i+1] = toks[i+2]
    return toks


def add_processed_keys(para, verbose=False):
    """ Add keys to para: 
    text_final (currently (should be) same as flattened 'text'), 
    hyperlinks
    sentence_spans
    id_new: id with '_0' added
    status: '' or 'nm' if new text doesnt match current text or 'bt' if not text (or 'btnm' for both)
    """
    # identify hyperlinks, anchor text spans and remove links from text_with_links s.t. should = text
    tl = ''.join(para['text_with_links'])
    toks = utils.flatten(para['charoffset_with_links']) #[start1, end1, start2, end2,...]
    toks = adj_token_boundaries(toks)
    currtext = ""
    startref=False
    anchor=""
    links_dict = {}
    hlink = None
    status = ''
    for i in range(0, len(toks), 2):  #i=start, i+1=end
        if verbose:
            print(i, toks[i], toks[i+1], tl[toks[i]: toks[i+1]])
        if tl[toks[i]:toks[i+1]].startswith('<a href="'):
            astart = len(currtext)
            hlink = unescape(unquote(tl[toks[i]+9:toks[i+1]-2])).strip()
            if hlink == '':
                print(f"id:{para['id']}: blank href. skipping and creating without hyperlinks.")
                status += 'blankhref_'
                break
            if not tl[toks[i]:toks[i+1]].endswith('">'):
                endref = tl[toks[i]:toks[i+1]].find('">')
                if endref == -1:
                    print(f"id:{para['id']}: bad href. skipping and creating without hyperlinks.")
                    status += 'badhref_'
                    break
                currtext += tl[toks[i]:toks[i+1]][endref+2:]
            startref=True
        elif tl[toks[i]:toks[i+1]].startswith('</a>'):
            startref=False
            aend = len(currtext)
            currtext += tl[toks[i]:toks[i+1]][4:]  # '' or ' '
            if hlink is None:
                print(f"id:{para['id']}: No <a href before </a>. anchor={anchor}. Anchor skipped and creating without hyperlinks.")
                status += 'nohlink_'
                break
            else:    
                if links_dict.get(hlink) is None:
                    links_dict[hlink] = []
                links_dict[hlink].append( {"anchor_text": anchor.strip(), "span": [astart, aend]} )
            anchor=""
            hlink=None
        elif startref:
            anchor += tl[toks[i]:toks[i+1]]
            currtext += tl[toks[i]:toks[i+1]]
        else:
            currtext += tl[toks[i]:toks[i+1]]
    if status == '':
        txt = ''.join(para['text'])
        if currtext != txt:  #check processed text matches hpqa processed text
            if "<a href=" in txt:
                print(f"id:{para['id']}: mismatch and href in orig text. Skipping and creating without hyperlinks.")
                status += 'nm_hrefinorig_'
            else:    
                status += 'nm'
                print(f"Unmatched text. ID: {para['id']}")
        elif currtext.strip() == '':
            status = 'bt' + status
            print(f"Blank text. ID: {para['id']}")
    
        if verbose:
            print(f"currtext#{currtext}#")
            print(f"hpqatext#{''.join(para['text'])}#")
            for l in links_dict:
                print(f"linksdetail#ANCHOR:{links_dict[l][0]['anchor_text']}:ANCHORSPAN:{currtext[links_dict[l][0]['span'][0]:links_dict[l][0]['span'][1]]}#HLINK:{l}#")
    if status != '':  # tokenising error for hyperlinks or corrupt hyperlinks, just use text minus hyperlinks
        links_dict = {}
        currtext = ''.join(para['text'])                
    
    # identify sentence spans - use HPQA form - no trailing space on 1st sent then leading space for each after that.
    sentence_spans = []
    start = 0
    for s in para['text']:
        end = start + len(s)
        sentence_spans.append( [start, end] )
        start = end
    if verbose:
        for span in sentence_spans:
            print(f"span#{span}#{currtext[span[0]: span[1]]}#")
    para['text_final'] = currtext
    para['hyperlinks'] = links_dict
    para['sentence_spans'] = sentence_spans
    para['status'] = status
    para['id_new'] = para['id'] + '_0'
    return


def save_aiso(paras):
    """ save non-blank records """
    print('Formatting for output...')
    out = ['id\ttext\ttitle\thyperlinks\tsentence_spans\n']
    for para in paras:
        if para['text_final'].strip() != '':
            outstr = f"{para['id_new']}\t{para['text_final']}\t{para['title'].strip()}\t{json.dumps(para['hyperlinks'])}\t{json.dumps(para['sentence_spans'])}\n"
            out.append(outstr)
    print(f'Saving {AISO_FILE} ...')
    with open(AISO_FILE, 'w') as f:
        f.write(''.join(out))    
    return



#tstfile = os.path.join(INDIR_BASE, 'AA', 'wiki_00.bz2')
#content = load_bz2_to_jsonl(tstfile)

filelist = glob.glob(INDIR_BASE +'/*/wiki_*.bz2')
paras = []
for i, infile in enumerate(filelist):
    paras += load_bz2_to_jsonl(infile)
    if i % 500 == 0:
        print(f"Processed {i} of {len(filelist)}")
# ''.join(paras[x]['text'])
# see https://github.com/qipeng/golden-retriever/blob/master/scripts/index_processed_wiki.py


#utils.saveas_jsonl(paras, os.path.join(OUTDIR, 'hpqa_paras_combo_raw.jsonl')) # 5233329
#paratest = copy.deepcopy(paras[2])
#add_processed_keys(paratest)
#para = get_para(paras, idx='12780069')  #hebrew
#para = get_para(paras, idx='1917146')  # 1st tok didnt start at 0. Fixed
#para = get_para(paras, idx='1936838')  # </a> mismatch token doesntcontain full href IGNORE Hlinks
#para = get_para(paras, idx='1895365')  # </a> mismatch token doesntcontain full href IGNORE Hlinks
#para = get_para(paras, idx='1922788')  # %20 at start of href. fixed
#para = get_para(paras, idx='13064089')  # blank href. skip klinks
#para = get_para(paras, idx='356443')  # mismatch
#para = get_para(paras, idx='356454')  # mismatch
#para2=copy.deepcopy(para)
#add_processed_keys(para2, verbose=True)

# Create output data as extra keys
for i, para in enumerate(paras):
    add_processed_keys(para)    
    if i % 250000 == 0:
        print(f"Processed: {i}") # Outputting 5233235 non-blank paras of 5233329 originally..MATCHES MDR counts

cnt = 0
for para in paras:
    if para['text_final'].strip() != '':
        cnt += 1
print(f"Outputting {cnt} non-blank paras of {len(paras)} originally..")        
        
save_aiso(paras)


#NOTE: AISO removes invalid links in transition_dataset.py so not doing it here
#TODO check any other text cleanup AISO probably does...  

#TODO How to tell if AISO train/dev jsonl titles maps to this corpus?? - load and compare...


