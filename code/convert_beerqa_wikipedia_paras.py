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
import json
import bz2
import glob
from urllib.parse import unquote  #convert percent encoding eg %28%20%29 -> ( )   quote does opposite
from html import unescape
import copy


####### MDR:
import utils #Duplicate "utils" with AISO so must run MDR and AISO portions separately from different working directories

OUTDIR = '/data/thar011/gitrepos/compgen_mdr/data/hpqa_raw_tim'


mdr_hpqa = utils.loadas_json('/data/thar011/gitrepos/compgen_mdr/data/hotpot_index/wiki_id2doc.json') # 5233329

mdr_out = [{'title': v['title'], 'text': v['text']} for v in mdr_hpqa.values() if v['text'].strip() != ''] # 5233235 strips blanks

utils.saveas_jsonl(mdr_out, os.path.join(OUTDIR, 'hpqa_abstracts_tim.jsonl'))


#test
#datatest = [json.loads(l) for l in open(os.path.join(OUTDIR, 'hpqa_abstracts_tim.jsonl')).readlines()]


###### AISO
#INDIR_BASE = '/data/thar011/gitrepos/compgen_mdr/data/hpqa_raw_tim/enwiki-20171001-pages-meta-current-withlinks-abstracts'
INDIR_BASE = '/home/thar011/data/beerqa/enwiki-20200801-pages-articles-tokenized'
#AISO_FILE = '/data/thar011/gitrepos/AISO/data/corpus/hotpot-paragraph.strict.tjh_v2.tsv'
#AISO_DEV = '/data/thar011/gitrepos/AISO/data/hotpot-step-dev.strict.refined.jsonl'
#AISO_TRAIN = '/data/thar011/gitrepos/AISO/data/hotpot-step-train.strict.refined.jsonl'
AISO_FILE = '/data/thar011/gitrepos/AISO/data/corpus/beer.tsv'
BEER_DEV = '/home/thar011/data/beerqa/beerqa_dev_v1.0.json'
BEER_TRAIN = '/home/thar011/data/beerqa/beerqa_train_v1.0.json'

tstfile = '/AA/wiki_00.bz2'

def load_jsonl(file, verbose=True):
    """ Load a list of json msgs from a file formatted as 
           {json msg 1}
           {json msg 2}
           ...
    """
    if verbose:
        print('Loading json file: ', file)
    with open(file, "r") as f:
        all_json_list = f.read()
    all_json_list = all_json_list.split('\n')
    num_jsons = len(all_json_list)
    if verbose:
        print('JSON as text successfully loaded. Number of json messages in file is ', num_jsons)
    all_json_list = [json.loads(j) for j in all_json_list if j.strip() != '']
    if verbose:
        print('Text successfully converted to JSON.')
    return all_json_list

def flatten(alist):
    """ flatten a list of nested lists
    """
    t = []
    for i in alist:
        if not isinstance(i, list):
             t.append(i)
        else:
             t.extend(flatten(i))
    return t


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


def build_title_idx(paras):
    """ The HPQA corpus files contain some hrefs that have casing different to the actual title casing.
    Simply making everything lowercase wont work as this creates a smallish number of duplicates.
    So we build a dict with key title.lower():
        {'title_lower': [{'title': the title, 'id': id of this title}]}
    """
    titledict = {}
    dupdict = {}
    for para in paras:
        tlower = para['title'].lower()
        title_entry = {'title': para['title'], 'id':para['id']}
        if titledict.get(tlower) is None:
            titledict[tlower] = [title_entry]
        else:
            titledict[tlower].append(title_entry)
            print(f"Dup lowercase: {tlower}  New Entry:{title_entry}")
            dupdict[tlower] = titledict[tlower]
    return titledict, dupdict
            

def get_para(paras, idx=None, title=None):
    if idx is not None:
        out = [p for p in paras if p['id'] == idx]
    elif title is not None:
        out = [p for p in paras if p['title'] == title]
    return out[0]    


def print_para(para, keys=['id','title','text']):
    """ Print selected keys from a single para
    """
    outstr = ' '.join([k+': '+para.get(k) for k in keys])
    print(outstr)
    return
        

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


def add_processed_keys(para, titledict, verbose=False):
    """ Add keys to para: 
    text_final (currently (should be) same as flattened 'text'), 
    hyperlinks
    sentence_spans
    id_new: id with '_0' added
    status: '' or 'nm' if new text doesnt match current text or 'bt' if no text (or 'btnm' for both)
    """
    #preprocess to remove non-href tags etc that are inconsistent between text [] and text_with_links ['<templatestyles src="Refbegin/styles.css" /']
    text_w_links = copy.deepcopy(para['text_with_links'])
    offsets_w_links = copy.deepcopy(para['offsets_with_links'])
    for i,t in enumerate(para['text']):
        if t == '':
            text_w_links[i] = ''
            offsets_w_links[i] = []
              
    # identify hyperlinks, anchor text spans and remove links from text_with_links s.t. should = text
    tl = ''.join(text_w_links)  #''.join(para['text_with_links'])
    toks = flatten(offsets_w_links)   #flatten(para['offsets_with_links']) #[start1, end1, start2, end2,...] #HPQA was 'charoffset_with_links'
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
        para['hyperlinks_bad'] = links_dict
        para['text_final_bad'] = currtext
        links_dict = {}
        currtext = ''.join(para['text'])                

    # Map title casing
    links_dict_mapped = {}  
    if links_dict != {}:
        for l in links_dict:
            tlower = l.lower()
            tmap = titledict.get(tlower)
            if tmap is None:
                links_dict_mapped[l] = links_dict[l]  # href not in corpus, just copy existing
            else:
                if len(tmap) == 1:
                    hlink = tmap[0]['title'] # only one candidate, use that
                else:
                    hlink == ''
                    for t in tmap:
                        if l == t['title']: # exact match amongst candidates found, use that
                            hlink = l
                            break
                    if hlink == '':
                        hlink = tmap[0]['title']
                        print(f"id:{para['id']}: No exact match for href #{l}# casing found and multiple candidates. Assigning first one #{hlink}#")
                links_dict_mapped[hlink] = links_dict[l]
            
    
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
    para['hyperlinks'] = links_dict_mapped
    para['sentence_spans'] = sentence_spans
    para['status'] = status
    para['id_new'] = para['id'] + '_0'        
    return


def map_title_case(hlink, titledict):
    """ Some titles in HPQA abstracts have incorrect casing. Attempt to map casing.
    """
    tlower = hlink.lower()
    tmap = titledict.get(tlower)
    if tmap is not None:                # if not found at all just return existing hlink
        if len(tmap) == 1:
            hlink = tmap[0]['title']    # only one candidate, use that
        else:
            for t in tmap:
                if hlink == t['title']: # exact match amongst candidates found, use that
                    return hlink
            hlink = tmap[0]['title']    # otherwise just return first
    return hlink


def get_links(sent, sent_w_links, links_dict, titledict, sent_offset):
    """ Parse sent_w_links, extract and return link titles, anchors, anchor spans
    sent_offset is the length of the current paragraph before adding sent
    updates links_dict: {'Link title': [ {'anchor_text': 'some text', 'span': [startchar, endchar]} ] }
    """
    start_href = sent_w_links.find('<a href="')
    sent_ptr = 0
    while start_href != -1:
        start_href = start_href + 9
        end_href = sent_w_links.find('">')
        if end_href > start_href:
            hlink = unescape(unquote(sent_w_links[start_href:end_href])).strip()
            hlink = map_title_case(hlink, titledict)
            end_href += 2
            end_anchor = sent_w_links.find('</a>')
            if end_anchor > end_href:
                anchor_text = sent_w_links[end_href:end_anchor].strip()
                anchor_text_start = sent[sent_ptr:].find(anchor_text)
                if anchor_text_start > -1:
                    anchor_text_start += sent_ptr
                    sent_ptr = anchor_text_start + len(anchor_text)
                    if links_dict.get(hlink) is None:
                        links_dict[hlink] = []
                    links_dict[hlink].append( {"anchor_text": anchor_text, "span": [sent_offset+anchor_text_start, sent_offset+sent_ptr]} )
                    sent_w_links = sent_w_links[end_anchor+4:]
                else: # anchor text not found
                    sent_w_links = sent_w_links[end_anchor+4:]
            else: # no </a> found
                sent_w_links = ''
        else: # couldnt find hlink termination str ">
            sent_w_links = ''
        start_href = sent_w_links.find('<a href="')
    return
    

def process_doc(doc, titledict, verbose=False):
    """ Split doc into paras and process each para
    doc: {"id": 12, "url": "https://en.wikipedia.org/wiki?curid=12",
    "title": "Anarchism",
    "text": ["Anarchism", "\n\nAnarchism is a political philosophy and movement that rejects all involuntary ...],
    "offsets": [[[0, 9]], [[11, 20], [21, 23], [24, 25], [26, 35], [36, 46], [47, 50], [51, 59], [60, 64], [65, 72], [73, 76], ...],
    "text_with_links": ["Anarchism", "\n\nAnarchism is a <a href=\"political%20philosophy\">political philosophy</a> and <a href=\"Political%20movement\">movement</a> that rejects ...],
    "offsets_with_links": [[[0, 9]], [[11, 20], [21, 23], [24, 25], [26, 59], [59, 68], [69, 79], [79, 83], [84, 87], [88, 119], [119, 127], [127, 131], [132, 136], [137, 144], ...] }
    
    len(text) = len(text_with_links)
                  
    """
    doc['paras'] = []
    #text_w_links = copy.deepcopy(para['text_with_links'])
    curr_para = ''
    links_dict = {}
    sentence_spans = []
    curr_para_idx = 0
    for i, sent in enumerate(doc['text']):  # for each sentence
        curr_sent = sent.replace('\n', '')
        if not sent.lstrip(' ').startswith('\n'):   # not start of new para
            curr_para_len = len(curr_para)
            get_links(curr_sent, doc['text_with_links'][i], links_dict, titledict, curr_para_len)
            sentence_spans.append( [curr_para_len, curr_para_len+len(curr_sent)] )
            curr_para += curr_sent
        else:                                       # start of new para
            if len(curr_para.strip()) > 0 and len(curr_para.split()) > 8:  # following https://github.com/beerqa/IRRR/blob/main/scripts/index_processed_wiki.py ignore headings
                doc['paras'].append( {'pid': str(curr_para_idx),  'text':curr_para, 'hyperlinks':copy.deepcopy(links_dict), 'sentence_spans': copy.deepcopy(sentence_spans) } )
            links_dict = {}
            get_links(curr_sent, doc['text_with_links'][i], links_dict, titledict, 0)
            sentence_spans = [ [0, len(curr_sent)] ]
            curr_para = curr_sent
            curr_para_idx += 1
    if curr_para != '':
        doc['paras'].append( {'pid': str(curr_para_idx), 'text':curr_para, 'hyperlinks':copy.deepcopy(links_dict), 'sentence_spans': copy.deepcopy(sentence_spans) } )
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



tstfile = os.path.join(INDIR_BASE, 'AA', 'wiki_00.bz2')
content = load_bz2_to_jsonl(tstfile)  #list
print(len(content))  #29
print(content[0].keys()) # dict_keys(['id', 'url', 'title', 'text', 'offsets', 'text_with_links', 'offsets_with_links'])
print(len(''.join(content[0].get('text')).split('\n\n'))) #56
paratest = copy.deepcopy(content)
titledicttest, dupdicttest = build_title_idx(paratest) # 2607 dups
for i, doc in enumerate(paratest):
    print('Processing:', i)
    process_doc(doc, titledicttest, verbose=False)
    #add_processed_keys(p, titledicttest, verbose=False)

print(paratest[0].keys()) # dict_keys(['id', 'url', 'title', 'text', 'offsets', 'text_with_links', 'offsets_with_links', 'paras'])
print(paratest[0]['status']) #nm



filelist = glob.glob(INDIR_BASE +'/*/wiki_*.bz2')
paras = []
for i, infile in enumerate(filelist):
    paras += load_bz2_to_jsonl(infile)
    if i % 500 == 0:
        print(f"Processed {i} of {len(filelist)}")
# ''.join(paras[x]['text'])
# see https://github.com/qipeng/golden-retriever/blob/master/scripts/index_processed_wiki.py

titledict, dupdict = build_title_idx(paras) # 2607 dups

dupkeys = list(dupdict.keys())


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
#add_processed_keys(para2, titledict, verbose=True)

# Create output data as extra keys
for i, para in enumerate(paras):
    add_processed_keys(para, titledict)    
    if i % 250000 == 0:
        print(f"Processed: {i}") # Outputting 5233235 non-blank paras of 5233329 originally..MATCHES MDR counts

cnt = 0
for para in paras:
    if para['text_final'].strip() != '':
        cnt += 1
print(f"Outputting {cnt} non-blank paras of {len(paras)} originally..")   # Outputting 5233235 non-blank paras of 5233329 originally.. MATCHES MDR counts    
        
save_aiso(paras)


#NOTE: AISO removes invalid links in transition_dataset.py so not doing it here
#TODO check any other text cleanup AISO probably does...  

#TODO How to tell if AISO train/dev jsonl titles maps to this corpus?? - load and compare...

# working dir /data/thar011/gitrepos/AISO
import utils.data_utils

corpus, title2id = utils.data_utils.load_corpus(AISO_FILE, for_hotpot=True, require_hyperlinks=True)
corpus['12_0']  
# {'title': 'Anarchism', 'text': 'Anarchism is a political philosophy that advocates self-governed societies based on voluntary institutions. These are often described as stateless societies, although several authors have defined them more specifically as institutions based on non-hierarchical free associations. Anarchism holds the state to be undesirable, unnecessary and harmful.',
# 'sentence_spans': [(0, 107), (107, 279), (279, 349)],
# 'hyperlinks': {'Political philosophy': [(15, 35)],  'Self-governance': [(51, 64)],  'Stateless society': [(137, 156)],  'Hierarchy': [(248, 260)],  'Free association (communism and anarchism)': [(261, 278)],  'State (polity)': [(300, 305)]}}
title2id['Anarchism'] # '12_0'

title2id['Stateless society']
title2id['Political philosophy']

len(title2id)  # 5233235

#tlower = list(title2id.keys())
#tlower = [t.lower() for t in tlower]
#print(f"correct Len:{len(title2id)}  LOWER len:{len(set(tlower))}")  # correct Len:5233235  LOWER len:5230617

def check_match(corpus, title2id, sample, verbose=False):
    """ Check AISO question title and id links match corpus
    {'_id': '5a8b57f25542995d1e6f1371', 'question': 'Were Scott Derrickson and Ed Wood of the same nationality?', 'answer': 'yes',
     'sp_facts': {'Scott Derrickson': [0], 'Ed Wood': [0]},
     'hard_negs': ['528464_0',  '39514096_0',  '22418962_0',  '7102461_0',  '3745444_0',  '41668588_0',  '18110_0',  '36528221_0',  '2571604_0',  '27306717_0'],
     'hn_scores': [0.0018630551639944315,  0.0005404593539424241,  0.000537772080861032,  0.0005160804139450192,  0.0005122957518324256,  0.0005065873847343028,  0.0005059774266555905,  0.0004960809019394219,  0.0004910272546112537,  0.0004873361031059176],
     'state2action': {'initial': {'query': 'Scott Derrickson',  'action': 'MDR',
                                   'sp_ranks': {'BM25': {'2816539_0': 0, '10520_0': 2000},
                                                'BM25+Link': {'2816539_0': 1, '10520_0': 2000},
                                                'MDR': {'2816539_0': 1, '10520_0': 0},
                                                'MDR+Link': {'2816539_0': 2000, '10520_0': 34}}},
                      'Scott Derrickson': {'query': 'Ed Wood',   'action': 'BM25',   'sp2_ranks': {'BM25': 0, 'BM25+Link': 1, 'MDR': 0, 'MDR+Link': 53}},
                      'Ed Wood': {'query': 'Ed Wood',   'action': 'MDR',   'sp2_ranks': {'BM25': 2000,    'BM25+Link': 2000,    'MDR': 0,    'MDR+Link': 2000}}}}
    """
    sp_facts_bad = []
    for k in sample['sp_facts'].keys():
        if title2id.get(k) is None:
            sp_facts_bad.append(k)
            if verbose:
                print(f"{sample['_id']}: sp_facts: Can't find: {k}")
    hard_negs_bad = []
    for n in sample['hard_negs']:
        if corpus.get(n) is None:
            hard_negs_bad.append(n)
            if verbose:
                print(f"{sample['_id']}: hard_negs: Can't find: {n}")
    s2a_bad = {}
    for k in sample['state2action']['initial']['sp_ranks'].keys():
        s2a_bad[k] = []
        for n in sample['state2action']['initial']['sp_ranks'][k].keys(): 
            if corpus.get(n) is None:
                s2a_bad[k].append(n)
                if verbose:
                    print(f"{sample['_id']}: state2action initial {k}: Can't find: {n}")
    out = {'sp_facts_bad':sp_facts_bad, 'hard_negs_bad': hard_negs_bad, 's2a_bad': s2a_bad}
    return out


def check_all(corpus, title2id, hpqa_split, verbose=True):
    log = []
    sp_facts_count = 0
    hn_count = 0
    s2a_count = {'BM25':0, 'BM25+Link':0, 'MDR':0, 'MDR+Link':0}
    for s in hpqa_split:
        log.append( check_match(corpus, title2id, s, verbose=verbose) )
        if len(log[-1]['sp_facts_bad']) > 0:
            sp_facts_count += 1
        if len(log[-1]['hard_negs_bad']) > 0:
            hn_count += 1
        for k in ['BM25', 'BM25+Link', 'MDR', 'MDR+Link']:
            if len(log[-1]['s2a_bad'][k]) > 0:
                s2a_count[k] += 1
            
    print(f"Finished: Total:{len(log)}  Bad sp_facts:{sp_facts_count}  Bad hn: {hn_count}  Bad s2a initial: {s2a_count}") 
    return log


aiso_dev = load_jsonl(AISO_DEV)
#check_match(corpus, title2id, aiso_dev[1119], verbose=True)
log_dev = check_all(corpus, title2id, aiso_dev, verbose=False)  # Finished: Total:7405  Bad sp_facts:0  Bad hn: 134  Bad s2a initial: {'BM25': 11, 'BM25+Link': 11, 'MDR': 11, 'MDR+Link': 11}

aiso_train = load_jsonl(AISO_TRAIN)
log_train = check_all(corpus, title2id, aiso_train, verbose=False)  # Finished: Total:90447  Bad sp_facts:0  Bad hn: 1766  Bad s2a initial: {'BM25': 123, 'BM25+Link': 123, 'MDR': 123, 'MDR+Link': 123}




