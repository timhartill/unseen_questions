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
from urllib.parse import unquote, quote  #convert percent encoding eg %28%20%29 -> ( )   quote does opposite
from html import unescape, escape
import copy

import text_processing
import utils_elasticsearch as UES

####### MDR:
#import utils #Duplicate "utils" with AISO so must run MDR and AISO portions separately from different working directories

#OUTDIR = '/data/thar011/gitrepos/compgen_mdr/data/hpqa_raw_tim'


#mdr_hpqa = utils.loadas_json('/data/thar011/gitrepos/compgen_mdr/data/hotpot_index/wiki_id2doc.json') # 5233329
mdr_hpqa = json.load(open('/data/thar011/gitrepos/compgen_mdr/data/hotpot_index/wiki_id2doc.json'))

#mdr_out = [{'title': v['title'], 'text': v['text']} for v in mdr_hpqa.values() if v['text'].strip() != ''] # 5233235 strips blanks

#utils.saveas_jsonl(mdr_out, os.path.join(OUTDIR, 'hpqa_abstracts_tim.jsonl'))


#test
#datatest = [json.loads(l) for l in open(os.path.join(OUTDIR, 'hpqa_abstracts_tim.jsonl')).readlines()]


###### AISO
#INDIR_BASE = '/data/thar011/gitrepos/compgen_mdr/data/hpqa_raw_tim/enwiki-20171001-pages-meta-current-withlinks-abstracts'
INDIR_BASE = '/home/thar011/data/beerqa/enwiki-20200801-pages-articles-tokenized'
#AISO_FILE = '/data/thar011/gitrepos/AISO/data/corpus/hotpot-paragraph.strict.tjh_v2.tsv'
#AISO_DEV = '/data/thar011/gitrepos/AISO/data/hotpot-step-dev.strict.refined.jsonl'
#AISO_TRAIN = '/data/thar011/gitrepos/AISO/data/hotpot-step-train.strict.refined.jsonl'
AISO_FILE = '/data/thar011/gitrepos/AISO/data/corpus/beer_v1.tsv'
BEER_WIKI_SAVE = '/home/thar011/data/beerqa/enwiki-20200801-pages-articles-compgen.json'
BEER_TITLE_SAVE = '/home/thar011/data/beerqa/enwiki-20200801-titledict-compgen.json'
BEER_DEV = '/home/thar011/data/beerqa/beerqa_dev_v1.0.json'
BEER_TRAIN = '/home/thar011/data/beerqa/beerqa_train_v1.0.json'

MDR_DEV = '/data/thar011/gitrepos/compgen_mdr/data/hotpot/hotpot_dev_with_neg_v0.json'
MDR_TRAIN = '/data/thar011/gitrepos/compgen_mdr/data/hotpot/hotpot_train_with_neg_v0.json'

ES_INDEX = 'enwiki-20200801-paras-v1'




#tstfile = '/AA/wiki_00.bz2'

def saveas_json(obj, file, mode="w", indent=5, add_nl=False):
    """ Save python object as json to file
    default mode w = overwrite file
            mode a = append to file
    indent = None: all json on one line
                0: pretty print with newlines between keys
                1+: pretty print with that indent level
    add_nl = True: Add a newline before outputting json. ie if mode=a typically indent=None and add_nl=True   
    Example For outputting .jsonl (note first line doesn't add a newline before):
        saveas_json(pararules_sample, DATA_DIR+'test_output.jsonl', mode='a', indent=None, add_nl=False)
        saveas_json(pararules_sample, DATA_DIR+'test_output.jsonl', mode='a', indent=None, add_nl=True)
          
    """
    with open(file, mode) as fp:
        if add_nl:
            fp.write('\n')
        json.dump(obj, fp, indent=indent)
    return True    


def saveas_jsonl(obj_list, file, initial_mode = 'w', verbose=True, update=5000):
    """ Save a list of json msgs as a .jsonl file of form:
        {json msg 1}
        {json msg 2}
        ...
        To overwrite exiting file use default initial_mode = 'w'. 
        To append to existing file set initial_mode = 'a'
    """
    if initial_mode == 'w':
        if verbose:
            print('Creating new file:', file)
        add_nl = False
    else:
        if verbose:
            print('Appending to file:', file)
        add_nl = True
    mode = initial_mode
    for i, json_obj in enumerate(obj_list):
            saveas_json(json_obj, file, mode=mode, indent=None, add_nl=add_nl)
            add_nl = True
            mode = 'a'
            if verbose:
                if i > 0 and i % update == 0:
                    print('Processed:', i)
    if verbose:
        print('Finished adding to:', file)        
    return True


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


def load_bz2_to_jsonl(infile, delkeys = ['offsets', 'offsets_with_links', 'url'], verbose=False):
    source_file = bz2.BZ2File(infile, "r")
    out = []
    count = 0
    for line in source_file:
        count += 1
        doc = json.loads(line)
        for k in delkeys:
            del doc[k]
        out.append(doc)
    source_file.close()
    if verbose:
        print(f"Read {count} rows from {infile}")
    return out


def build_title_idx(docs):
    """ The HPQA corpus files contain some hrefs that have casing different to the actual title casing.
    Simply making everything lowercase wont work as this creates a smallish number of duplicates.
    So we build a dict with key title.lower():
        {'title_lower': [{'title': the title, 'id': id of this title}]}
        Also title_lower is UNESCAPED meaning " ' & < > are as here not escaped like the orig titles which have eg &amp; for &
    """
    titledict = {}
    dupdict = {}
    for i, doc in enumerate(docs):
        tlower = unescape(doc['title'].lower()) # unescaped
        title_entry = {'title': doc['title'], 'id':doc['id'], 'idx': i} # not escaped so matches title in corpus
        if titledict.get(tlower) is None:
            titledict[tlower] = [title_entry]
        else:
            titledict[tlower].append(title_entry)
            print(f"Dup lowercase: {tlower}  New Entry:{title_entry}")
            dupdict[tlower] = titledict[tlower]
        if i % 1000000 == 0:
            print(f"Processed: {i} Dups so far:{len(dupdict)}")
    print(f"Total dups: {len(dupdict)}")
    return titledict, dupdict
            

def get_para(paras, idx=None, title=None):
    if idx is not None:
        out = [p for p in paras if p['id'] == idx]
    elif title is not None:
        out = [p for p in paras if p['title'] == title]
    return out[0]    


def map_title_case(hlink, titledict, verbose=False):
    """ Some titles in HPQA abstracts have incorrect casing. Attempt to map casing.
    Note unescape will map eg &amp; to & but will have no effect on already unescaped text so can pass either escaped or unescaped version
    """
    tlower = unescape(hlink.lower())
    tmap = titledict.get(tlower)
    status = 'nf'
    idx = -1
    if tmap is not None:                # if not found at all just return existing hlink
        if len(tmap) == 1:
            if hlink != tmap[0]['title']:
                if verbose:
                    print(f"Hlink case different. Orig:{hlink} actual: {tmap[0]['title']}")
                status = 'sc'
            else:
                status = 'sok'
            hlink = tmap[0]['title']    # only one candidate, use that
            idx = tmap[0]['idx']
        else:
            for t in tmap:
                if hlink == t['title']: # exact match amongst candidates found, use that
                    return hlink, 'mok', t['idx']
            hlink = tmap[0]['title']    # otherwise just return first
            idx = tmap[0]['idx']
            status = 'mc'
            if verbose:
                print(f"Hlink lower:{tlower} No exact match found so assigning first: {hlink}")
    return hlink, status, idx


def count_title_status(docs, titledict):
    """ Count title matching status and update hyperlinks into new 'hyperlinks_cased' key """
    count_dict = {'nf':0, 'sc':0, 'sok':0, 'mok':0, 'mc':0}
    para_count = 0
    for i, doc in enumerate(docs):
        for p in doc['paras']:
            para_count += 1
            new_links_dict = {}
            for h in p['hyperlinks'].keys():
                new_hlink, status, idx = map_title_case(h, titledict)
                count_dict[status] += 1
                if new_links_dict.get(new_hlink) is None:
                    new_links_dict[new_hlink] = p['hyperlinks'][h]
                else:
                    new_links_dict[new_hlink].extend(p['hyperlinks'][h])
            p['hyperlinks_cased'] = copy.deepcopy(new_links_dict)
        if i % 1000000 == 0:
            print(f"Processed: {i} docs {para_count} paras. Hyperlink counts: {count_dict}")
    print(f"Finished counting titles. {para_count} paras. Counts: {count_dict}")
    return count_dict


def cleanup(docs):
    """ Final cleanup """
    for i, doc in enumerate(docs):
        for p in doc['paras']:
            p['text'] = p['text'].replace('\t', ' ')
            del p['hyperlinks']
        if i % 1000000 == 0:
            print(f"Processed: {i} docs")
    print("Finished cleanup")
    return


def get_links(para, para_w_links, links_dict, verbose=False):
    """ Parse para_w_links, extract and return link titles, anchors, anchor spans
    updates links_dict: {'Link title': [ {'anchor_text': 'some text', 'span': [startchar, endchar]} ] }
    """
    start_href = para_w_links.find('<a href="')
    sent_ptr = 0
    while start_href != -1:
        start_href = start_href + 9
        end_href = para_w_links.find('">')
        if end_href > start_href:
            hlink = unquote(para_w_links[start_href:end_href]).strip()  # removed unescape(unquote(...))
            end_href += 2
            end_anchor = para_w_links.find('</a>')
            if end_anchor > end_href:
                anchor_text = para_w_links[end_href:end_anchor].strip()
                anchor_text_start = para[sent_ptr:].find(anchor_text)
                if anchor_text_start > -1:
                    anchor_text_start += sent_ptr
                    sent_ptr = anchor_text_start + len(anchor_text)
                    if links_dict.get(hlink) is None:
                        links_dict[hlink] = []
                    links_dict[hlink].append( {"anchor_text": anchor_text, "span": [anchor_text_start, sent_ptr]} )
                    para_w_links = para_w_links[end_anchor+4:]
                else: # anchor text not found
                    if verbose:
                        print(f'Anchor text #{anchor_text}# not found in para: {para[sent_ptr:]}')
                    para_w_links = para_w_links[end_anchor+4:]
            elif end_anchor == end_href:  # some hrefs dont have anchor text - skip them. eg <a href="http%3A//content.inflibnet.ac.in/data-server/eacharya-documents/5717528c8ae36ce69422587d_INFIEP_304/66/ET/304-66-ET-V1-S1__file1.pdf"></a>
                para_w_links = para_w_links[end_anchor+4:]
            else: # no </a> found
                if verbose:
                    print(f'</a> not found for hlink:{hlink}: {para_w_links}')
                para_w_links = ''
        else: # couldnt find hlink termination str ">
            if verbose:
                print(f'href termination "> not found: {para_w_links}')
            para_w_links = ''
        start_href = para_w_links.find('<a href="')
    return
    

def process_doc(doc, delkeys = ['text', 'text_with_links'], verbose=False):
    """ Split doc into paras and process each para. Paras with <= 8 words are skipped/
    doc: {"id": 12, "url": "https://en.wikipedia.org/wiki?curid=12",
    "title": "Anarchism",
    "text": ["Anarchism", "\n\nAnarchism is a political philosophy and movement that rejects all involuntary ...],
    "offsets": [[[0, 9]], [[11, 20], [21, 23], [24, 25], [26, 35], [36, 46], [47, 50], [51, 59], [60, 64], [65, 72], [73, 76], ...],
    "text_with_links": ["Anarchism", "\n\nAnarchism is a <a href=\"political%20philosophy\">political philosophy</a> and <a href=\"Political%20movement\">movement</a> that rejects ...],
    "offsets_with_links": [[[0, 9]], [[11, 20], [21, 23], [24, 25], [26, 59], [59, 68], [69, 79], [79, 83], [84, 87], [88, 119], [119, 127], [127, 131], [132, 136], [137, 144], ...] }
    
    Note: len(text) = len(text_with_links) = num sentences
                  
    """
    doc['paras'] = []
    curr_para = ''
    curr_para_w_links = ''
    links_dict = {}
    sentence_spans = []
    curr_para_idx = 0
    for i, sent in enumerate(doc['text']):  # for each sentence
        curr_sent = sent.replace('\n', '')
        curr_sent_w_links = doc['text_with_links'][i].replace('\n', '')
        if not sent.lstrip(' ').startswith('\n'):   # not start of new para
            curr_para_len = len(curr_para)
            sentence_spans.append( [curr_para_len, curr_para_len+len(curr_sent)] )
            curr_para += curr_sent
            curr_para_w_links += curr_sent_w_links
        else:                                       # start of new para
            if len(curr_para.strip()) > 0 and len(curr_para.split()) > 8:  # following https://github.com/beerqa/IRRR/blob/main/scripts/index_processed_wiki.py ignore headings
                get_links(curr_para, curr_para_w_links, links_dict, verbose=verbose)
                doc['paras'].append( {'pid': str(curr_para_idx),  'text':curr_para, 'hyperlinks':copy.deepcopy(links_dict), 'sentence_spans': copy.deepcopy(sentence_spans) } )
            links_dict = {}
            sentence_spans = [ [0, len(curr_sent)] ]
            curr_para = curr_sent
            curr_para_w_links = curr_sent_w_links
            curr_para_idx += 1
    if len(curr_para.strip()) > 0 and len(curr_para.split()) > 8:
        get_links(curr_para, curr_para_w_links, links_dict, verbose=verbose)
        doc['paras'].append( {'pid': str(curr_para_idx), 'text':curr_para, 'hyperlinks':copy.deepcopy(links_dict), 'sentence_spans': copy.deepcopy(sentence_spans) } )
    for k in delkeys:
        del doc[k]        
    return
    

def save_aiso(docs):
    """ save non-blank records """
    print(f'Outputting to {AISO_FILE}...')
    with open(AISO_FILE, 'w') as f:
        f.write('id\ttext\ttitle\thyperlinks\tsentence_spans\n')    
        for i, doc in enumerate(docs):
            for j, para in enumerate(doc['paras']): # make paras 0 based and contiguous
                newid = doc['id'] + '_' + str(j)
                outstr = f"{newid}\t{para['text']}\t{doc['title'].strip()}\t{json.dumps(para['hyperlinks_cased'])}\t{json.dumps(para['sentence_spans'])}\n"
                f.write(outstr)
            if i % 250000 == 0:
                print(f"Processed: {i}")
    print(f'Saved {AISO_FILE}')
    return



tstfile = os.path.join(INDIR_BASE, 'AA', 'wiki_00.bz2')
content = load_bz2_to_jsonl(tstfile)  #list
print(len(content))  #29
print(content[0].keys()) # dict_keys(['id', 'url', 'title', 'text', 'offsets', 'text_with_links', 'offsets_with_links'])
print(len(''.join(content[0].get('text')).split('\n\n'))) #56
paratest = copy.deepcopy(content)
titledicttest, dupdicttest = build_title_idx(paratest) # 
for i, doc in enumerate(paratest):
    print('Processing:', i)
    process_doc(doc, verbose=True)
    #inp = input('Press <enter>')
count_title_status(paratest, titledicttest)

print(paratest[0].keys()) # dict_keys(['id', 'url', 'title', 'text', 'offsets', 'text_with_links', 'offsets_with_links', 'paras'])
# 'paras' eg: 
#    {'pid': '1',
# 'text': 'Alchemy (from Arabic: "al-kīmiyā") is an ancient branch of natural philosophy, a philosophical and protoscientific tradition practiced throughout Europe, Africa, and Asia, originating in Greco-Roman Egypt in the first few centuries CE.',
# 'hyperlinks': {'Arabic': [{'anchor_text': 'Arabic', 'span': [14, 20]}],
#  'natural philosophy': [{'anchor_text': 'natural philosophy',
#    'span': [59, 77]}],
#  'philosophical': [{'anchor_text': 'philosophical', 'span': [81, 94]}],
#  'protoscience': [{'anchor_text': 'protoscientific', 'span': [99, 114]}],
#  'Egypt (Roman province)': [{'anchor_text': 'Greco-Roman Egypt', 'span': [187, 204]}]},
# 'sentence_spans': [[0, 235]]}


filelist = glob.glob(INDIR_BASE +'/*/wiki_*.bz2')  #18013 bz2 files
docs = []
for i, infile in enumerate(filelist):
    currdocs = load_bz2_to_jsonl(infile)
    for j, doc in enumerate(currdocs):
        process_doc(doc, verbose=False)    # Create output data as extra keys and delete orig keys
    docs += currdocs
    if i % 500 == 0:
        print(f"Processed {i} of {len(filelist)}")
print(f'Finished loading and processing! Docs: {len(docs)}')  # 6133150 docs
print('Building title dictionary...')
titledict, dupdict = build_title_idx(docs) #  6020 dups
dupkeys = list(dupdict.keys())

# map titles: adds 'hyperlinks_cased' key to docs paras
h_counts = count_title_status(docs, titledict) 
# Finished counting titles. 35706771 paras. Counts: {'nf': 19367786, 'sc': 14077482, 'sok': 68807203, 'mok': 294440, 'mc': 326587}
cleanup(docs) # replace \t in para text (just in case there were any) and del hyperlinks key

saveas_json(docs, BEER_WIKI_SAVE, indent=None)
saveas_json(titledict, BEER_TITLE_SAVE, indent=None)

#output AISO corpus file
save_aiso(docs)




#TODO output MDR
#TODO output GR
# see https://github.com/qipeng/golden-retriever/blob/master/scripts/index_processed_wiki.py



###### BeerQA train/dev exploration
#TODO Solve multi-para issue with BeerQA HPQA train/dev samples - choose 1 or concat both?
#TODO figure out training ordering in any case. See if can deduce the "correct" para?

# ensure titles in BeerQA trian/dev samples all appear in wiki titles - CONFIRMED
# do paras in BeerQa samples match corpus paras? YES but some bqa paras are substrings and/or difft casing and/or difft strip to corpus paras

mdr_dev = load_jsonl(MDR_DEV)  # obtain hpqa question type and neg paras from mdr
mdr_train = load_jsonl(MDR_TRAIN)
mdr_dev_q_idx = {m['question'].strip().lower():i for i, m in enumerate(mdr_dev)}
mdr_train_q_idx = {m['question'].strip().lower():i for i, m in enumerate(mdr_train)}

print('Loading corpus file..')
docs = json.load(open(BEER_WIKI_SAVE))
print('Loading title 2 dict file..')
titledict = json.load(open(BEER_TITLE_SAVE))
print("finished loading!")

beer_dev = json.load(open(BEER_DEV)) # dict_keys(['version', 'split', 'data'])
beer_train = json.load(open(BEER_TRAIN))

beer_dev['version'] # 1.0
beer_dev['split'] # 'dev'  beer_train split = 'train'
len(beer_dev['data']) # 14121 train= 134043
print(beer_dev['data'][0]) # {'id': 'b3d50a40b29d4283609de1d3f426aebce198a0b2', 'src': 'hotpotqa', 
# 'answers': ['Eschscholzia'], 
#'question': 'Which genus contains more species, Ortegocactus or Eschscholzia?', 
# 'context': [['Eschscholzia', 'Eschscholzia is a genus of 12 annual or perennial plants in the Papaveraceae (poppy) family. The genus was named after the Baltic German/Imperial Russian botanist Johann Friedrich von Eschscholtz (1793-1831). All species are native to Mexico or the southern United States.'], 
#             ['Eschscholzia', 'Leaves are deeply cut, glabrous and glaucous, mostly basal, though a few grow on the stem.'], 
#             ['Ortegocactus', 'Ortegocactus macdougallii is a species of cactus and the sole species of the genus Ortegocactus. The plant has a greenish-gray epidermis and black spines. It is only known from Oaxaca, Mexico.']]}
# add 'map': [ {'d_idx':123, 'p_idx': 4}, ... ] |map| = |context| :
# 'map': [{'d_idx': 4070176, 'p_idx': 0},  {'d_idx': 4070176, 'p_idx': 1},  {'d_idx': 2614908, 'p_idx': 0}]}

#map_title_case('Ortegocactus', titledict, verbose=False)


def match_para(para, doc_idx, docs, preproc=True):
    """ Attempt to match a sample paragraph with a particular paragraph in a corpus doc with title "title".
    Return idx of matching para in [paras] or -1
    """
    if doc_idx == -1:
        return -1
    if preproc:
        para = para.strip().lower()
    p_idx = -1
    for i, p in enumerate(docs[doc_idx]['paras']):
        if preproc:
            ptext = p['text'].strip().lower()
        else:
            ptext = p['text']
        if para in ptext: # ptext.startswith(para):  #a few beerqa paras are truncated, often after a ':' either ath the beginning or the end
            p_idx = i
            break
    return p_idx       
    

def check_beer_split(beer_split, titledict, docs):
    """ Check that title casing is correct for all samples ... """
    count_dict = {'nf':0, 'sc':0, 'sok':0, 'mok':0, 'mc':0, 'para_nf': 0}
    nf_list = []
    pnf_list = []
    for i, sample in enumerate(beer_split['data']):
        para_match = []
        for title, para in sample['context']:
            new_title, status, d_idx = map_title_case(title, titledict)
            count_dict[status] += 1
            if status == 'nf':
                nf_list.append( {'q_idx': i, 'title': title} )
            p_idx = match_para(para, d_idx, docs, preproc=True)
            para_match.append( {'d_idx': d_idx, 'p_idx': p_idx } )
            if p_idx == -1:
                count_dict['para_nf'] += 1
                pnf_list.append( {'q_idx':i, 'q_para': para, 'd_idx': d_idx} )
        sample['map'] = para_match
        if i % 50000 == 0:
            print(f"Processed: {i}  {count_dict}")
    print(f"Counts: {count_dict}")
    return count_dict, nf_list, pnf_list

#All train/dev titles found in titledict and can uniquely map each to a corpus title:
#About 90% of train/dev paras match corpus para exactly: not match: (train: 21967 of 236810, dev: 2473 of 22376)
#Over 99% match after strip().lower(): not match: train: 1477 dev: 147
# Even more match (train: 665 dev: 61) after look for corpus para.startswith(bqapara.strip.lower) since some bqa para truncated often at ':'
# finally, all match when look for bqapara.strip.lower in corpuspara.strip.lower
cd_train, nf_train, pnf_train = check_beer_split(beer_train, titledict, docs)  # Counts: {'nf': 0, 'sc': 986, 'sok': 234538, 'mok': 1286, 'mc': 0, 'para_nf': 0}
cd_dev, nf_dev, pnf_dev = check_beer_split(beer_dev, titledict, docs)  # Counts: {'nf': 0, 'sc': 81, 'sok': 22259, 'mok': 36, 'mc': 0, 'para_nf': 0}
#  {'idx': 13991, 'title': 'What\'s It Gonna Be (H "Two" O song)'}, titledict: "what's it gonna be (h &quot;two&quot; o song)"
#   {'idx': 14001, 'title': 'Merck & Co.'} in titledict: 'merck &amp; co.'  
# UNESCAPE(title) will make it map to beerqa dev/test
# I was UNESCAPING hyperlinks - need to not do that as titles are escaped. Just need to unquote them.
# make titledict key unescape(title.lower()) but keep the title entries as escaped to match corpus
# look up hlinks using unescape(hlink.lower) returning the escaped title
# look up beerqa titles using unescape(bqa.title.lower) - unescape should have no effect


def get_overlap(text_a, text_b, title_b, func='lcs'):
    """ Return Longest Common Subsequence between strings text_a and text_b and text_a and title_b
    text_a is assumed to be the query and punctuation in addition to stopwords are stripped from it following golden retriever
    text_b and title_b just has stopwords stripped from them.
    changing func-> 'lcss' returns longest common substring
    changing func-> 'isect' returns count of intersecting terms irrespective of order
    """
    text_a_toks = text_processing.word_tokenize(text_a) #nltk simple word tokenizer
    text_b_toks = text_processing.word_tokenize(text_b)
    title_b_toks = text_processing.word_tokenize(title_b)
    ta_toks, ta_idx  = text_processing.filter_stopwords2(text_a_toks)
    tb_toks, _ = text_processing.filter_stopwords(text_b_toks)
    ttlb_toks, _ = text_processing.filter_stopwords(title_b_toks)
    ta_toks = [t.lower() for t in ta_toks]
    tb_toks = [t.lower() for t in tb_toks]
    ttlb_toks = [t.lower() for t in ttlb_toks]
    if func == 'lcs':
        tb_size, tb_lcs, _ = text_processing.LCS(ta_toks, tb_toks)
        ttlb_size, ttlb_lcs, _ = text_processing.LCS(ta_toks, ttlb_toks)
    elif func == 'lcss':    
        tb_size, tb_lcs, _ = text_processing.LCSubStr(ta_toks, tb_toks)
        ttlb_size, ttlb_lcs, _ = text_processing.LCSubStr(ta_toks, ttlb_toks)
    else:
        tb_lcs = list(set(ta_toks).intersection( set(tb_toks) ))
        tb_size = len(tb_lcs)
        ttlb_lcs = list(set(ta_toks).intersection( set(ttlb_toks) ))
        ttlb_size = len(ttlb_lcs)
    if tb_size > ttlb_size:
        return tb_size, tb_lcs
    else:
        return ttlb_size, ttlb_lcs


def find_shortest_LCS(sample):
    """ Return the title of the para(s) with shortest LCS wrt question (if only 1, use as final para).
    """
    q = sample['question'].strip()
    shortest_lcs_len = 99999
    shortest_lcs_titles = []
    for title in sample['para_agg']:
        paras = ' '.join(sample['para_agg'][title])
        lcs_len, _ = get_overlap(q, paras, title)
        if lcs_len < shortest_lcs_len:
            shortest_lcs_titles = [title]
            shortest_lcs_len = lcs_len
        elif lcs_len == shortest_lcs_len:  # multiple paras have same lcs with q 
            shortest_lcs_titles.append(title)
    return shortest_lcs_titles            
    

def add_sequencing(beer_split, mdr_split, mdr_split_q_idx, titledict, docs):
    """ follow mdr paper sequencing algorithm and calculate paragraph sequencing for beerqa:
        if 'bridge': final "bridge" para is one mentioning the answer span. 
                     TODO if the answer span is in both, the one that has its title mentioned in the other passage is treated as the second.
                     TODO Use bqa dev/train doc/para mappings from above
    Merge question type and neg paras from mdr for HPQA
    Can simply concatenate paras with same title to make noisier samples or exclude paras+titles that don't contain the answer for cleaner samples
    Nb: All squad dev have exactly 1 para but 874 squad train have 2. 
    """
    nf_idx = []
    nf_mdr = []
    tot_hpqa = 0
    tot_hpqa_comp = 0
    count_dict = {'squad': {'comp':0, 'ans_0': 0, 'ans_1': 0, 'ans_2': 0, 'ans_3':0, 'ans_over_3': 0},
                  'hotpotqa': {'comp':0, 'ans_0': 0, 'ans_1': 0, 'ans_2': 0, 'ans_3':0, 'ans_over_3': 0},
                  'squad_unique_titles': {'comp':0, 'ans_0': 0, 'ans_1': 0, 'ans_2': 0, 'ans_3':0, 'ans_over_3': 0, 'ans_2_refine':{'tot': 0, 'nf': 0, 'got_1': 0, 'got_2': 0, 'got_2_anseqtitle':0, 'got_2_shortestlcstitle':0, 'nf_shortestlcstitle':0}},
                  'hotpotqa_unique_titles': {'comp':0, 'ans_0': 0, 'ans_1': 0, 'ans_2': 0, 'ans_3':0, 'ans_over_3': 0, 'ans_2_refine':{'tot': 0, 'nf': 0, 'got_1': 0, 'got_2': 0, 'got_2_anseqtitle':0, 'got_2_shortestlcstitle':0, 'nf_shortestlcstitle':0}},
                 }
    for i,sample in enumerate(beer_split['data']):
        sample['type'] = ''
        sample['neg_paras'] = []
        sample['mdr_bridge'] = None
        if sample['src'] == 'hotpotqa':
            tot_hpqa += 1
            mdr_idx = mdr_split_q_idx.get(sample['question'].strip().lower())
            if mdr_idx is not None: # combine info from hpqa and mdr
                sample['type'] = mdr_split[mdr_idx]['type']
                sample['neg_paras'] = mdr_split[mdr_idx]['neg_paras']
                sample['mdr_bridge'] = mdr_split[mdr_idx].get('bridge')
                if sample['type'] == 'comparison':
                    tot_hpqa_comp += 1
            else:
                nf_mdr.append(i)
                
        ans = sample['answers'][0].strip().lower()
        para_seq = []
        para_title = set()
        para_title_eq_ans = ''
        para_agg = {}  # {'title': [p1, p2]} Aggregate paras from same title for convenience
        for j, (title, para) in enumerate(sample['context']):  # This pass adds keys to assist in second pass below
            if para_agg.get(title) is None:
                para_agg[title] = [para]
            else:
                para_agg[title].append(para)
            if sample['src'] == 'hotpotqa' and sample['type'] == 'comparison': # For comp questions select paras as intermediate/final randomly
                continue
            if ans == title.strip().lower(): # if answer = title of para we will treat as the final para in case of both paras containing the answer
                para_title_eq_ans = title
            if ans in para.lower() or ans in title.lower():  # If answer anywhere in para
                para_seq.append(j)
                para_title.add(title)
        sample['para_agg'] = para_agg
        sample['para_title_eq_ans'] = para_title_eq_ans
        sample['para_has_ans'] = para_seq
        sample['para_ans_titles'] = para_title       # unique titles that have answer embedded in their passages and hence could be final (unless more than one of them in which case must select the one that referred to by to the other as final.)
        if len(para_title) == 0:
            if sample['src'] == 'squad' or (sample['src'] == 'hotpotqa' and sample['type'] != 'comparison'): #ans not in ['yes', 'no']):
                nf_idx.append(i)
        
    for i,sample in enumerate(beer_split['data']): # 2nd pass: Calculate the final paragraph title in key 'final' as [final title(s)]. Squad: always 1 title. HPQA: if final determinable then added as [final title] otherwise [final title 1, final title 2] and data loader expected to select randomly
        ans = sample['answers'][0].strip().lower()
        l = len(sample['para_has_ans'])
        num_titles = len(sample['para_ans_titles'])
        all_titles = list(sample['para_agg'].keys())
        sample['final'] = all_titles  # default to all titles ie 1 for squad, 2 for hpqa (left like this for all comparison questions and any bridge where can't determine final)
        
        # Not using individual paras, rather aggreating by title so this part not used:
        if sample['src'] == 'hotpotqa' and sample['type'] == 'comparison': # not used - only considering titles and will concat paras with same title
            count_dict[sample['src']]['comp'] += 1
        elif l == 0:
            count_dict[sample['src']]['ans_0'] += 1
        elif l == 1:
            count_dict[sample['src']]['ans_1'] += 1
        elif l == 2:
            count_dict[sample['src']]['ans_2'] += 1
        elif l == 3:
            count_dict[sample['src']]['ans_3'] += 1
        else:
            count_dict[sample['src']]['ans_over_3'] += 1
            
        if sample['src'] == 'hotpotqa' and sample['type'] == 'comparison':
            count_dict[sample['src']+'_unique_titles']['comp'] += 1  # hpqa + comp = no para order, select either randomly
        elif num_titles == 0:
            count_dict[sample['src']+'_unique_titles']['ans_0'] += 1 # Anything here = error (currently empty)
        elif num_titles == 1:
            count_dict[sample['src']+'_unique_titles']['ans_1'] += 1 # Bingo: Single title has ans in it => final para in seq
            sample['final'] = list(sample['para_ans_titles'])
        elif num_titles == 2:
            count_dict[sample['src']+'_unique_titles']['ans_2'] += 1 # all squad have 1 unique title so just use title reference to tiebreak hpqa final para here
            final_para_title = set()  # titles of which at least 1 para has a hyperlink pointing to it
            p1, p2 = sample['para_ans_titles']
            found = False
            for i, ans_title in enumerate([p1, p2]):  # try to find final para as that which the other para contains a hyperlink to.
                if i == 0:
                    othertitle = p2
                else:
                    othertitle = p1
                for j, (title, para) in enumerate(sample['context']):
                    if ans_title == title:  # if any para with title matching current title that has answer in it...
                        d_idx, p_idx = sample['map'][j].values()
                        for hlink_title in docs[d_idx]['paras'][p_idx]['hyperlinks_cased']:
                            if unescape(hlink_title.lower()) == unescape(othertitle.lower()):
                                final_para_title.add(othertitle)
                                found = True
                                break
                    if found:
                        break
                found = False
            sample['para_ans_titles_refined'] = final_para_title
            count_dict[sample['src']+'_unique_titles']['ans_2_refine']['tot'] += 1
            refined_final_count = len(final_para_title)
            if refined_final_count == 0:  # neither para links to the other, try additional heuristic
                count_dict[sample['src']+'_unique_titles']['ans_2_refine']['nf'] += 1  # Anything here = error
                shortest_lcs_titles = find_shortest_LCS(sample)
                if len(shortest_lcs_titles) == 1: # take title with shortest longest common seq with title as final (if equal return both, pick randomly in dataset)
                    count_dict[sample['src']+'_unique_titles']['ans_2_refine']['nf_shortestlcstitle'] += 1  
                    sample['final'] = shortest_lcs_titles
                sample['shortest_lcs_titles'] = shortest_lcs_titles
            elif refined_final_count == 1:  #Bingo, only 1 para links to the other
                count_dict[sample['src']+'_unique_titles']['ans_2_refine']['got_1'] += 1  # correctly identified single final para
                sample['final'] = list(final_para_title)
            else: # paras link to each other, try additional heuristics to tiebreak
                count_dict[sample['src']+'_unique_titles']['ans_2_refine']['got_2'] += 1  # Anything here = titles link to each other!
                if sample['para_title_eq_ans'] != '':  #if one para's title = answer, take that as the final
                    count_dict[sample['src']+'_unique_titles']['ans_2_refine']['got_2_anseqtitle'] += 1  
                    sample['final'] = [ sample['para_title_eq_ans'] ]
                else: # take title with shortest longest common seq with title as final (if equal return both, pick randomly in dataset)
                    shortest_lcs_titles = find_shortest_LCS(sample)
                    if len(shortest_lcs_titles) == 1:
                        count_dict[sample['src']+'_unique_titles']['ans_2_refine']['got_2_shortestlcstitle'] += 1
                        sample['final'] = shortest_lcs_titles
                    sample['shortest_lcs_titles'] = shortest_lcs_titles
        elif num_titles == 3:
            count_dict[sample['src']+'_unique_titles']['ans_3'] += 1  # Anything here = error (currently empty)
        else:
            count_dict[sample['src']+'_unique_titles']['ans_over_3'] += 1  # Anything here = error (empty)
    
    nfcnt = 0
    difft_mdr = 0
    for i,sample in enumerate(beer_split['data']):  #final check comparing to MDR for hpqa
        if sample['src'] == 'squad':
            if len(sample['final']) != 1:
                print(f"ERROR idx:{i} squad sample has {len(sample['final'])} final paras...should have 1")
        else:
            if sample['type'] == 'comparison':
                if len(sample['final']) != 2:
                    print(f"ERROR idx:{i} hpqa comp sample has {len(sample['final'])} final paras...should have 2") 
            else:
                if len(sample['final']) == 0:
                    print(f"ERROR idx:{i} hpqa bridge sample has {len(sample['final'])} final paras...should have 1 or 2")                   
                elif len(sample['final']) == 2:
                    nfcnt += 1
                elif sample['final'][0].strip().lower() != sample['mdr_bridge'].strip().lower():
                    difft_mdr += 1
    print(f"Total BeerQA HPQA: {tot_hpqa}  Comparison: {tot_hpqa_comp}  Bridge: {tot_hpqa-tot_hpqa_comp}")
    print(f"Bridge: Single id-ed but different to MDR: {difft_mdr} Indeterminable final: {nfcnt} Total difft to MDR: {difft_mdr+nfcnt} of {tot_hpqa-tot_hpqa_comp}")    
    #DEV: Total BeerQA HPQA: 5989  Comparison: 1278  Bridge: 4711
    #Bridge: Single id-ed but different to MDR: 284 Indeterminable final: 20 Total difft to MDR: 304 of 4711    # Single but different to MDR: 3444 Indeterminable final: 288 Total difft to MDR: 3732 of 90447
    #TRAIN: Total BeerQA HPQA: 74758  Comparison: 15162  Bridge: 59596
    #Bridge: Single id-ed but different to MDR: 3444 Indeterminable final: 288 Total difft to MDR: 3732 of 59596
    
    print(f"Results: {count_dict}")
    print(f"No answer span found: {len(nf_idx)}")
    print(f"Number of hpqa samples not matched to MDR: {len(nf_mdr)}")
    return count_dict, nf_idx, nf_mdr

cd_dev, nf_dev, nf_mdr_dev = add_sequencing(beer_dev, mdr_dev, mdr_dev_q_idx, titledict, docs)  # Results: {'squad': {'comp': 0, 'ans_0': 0, 'ans_1': 7970, 'ans_2': 162, 'ans_3': 0, 'ans_over_3': 0}, 'hotpotqa': {'comp': 1278, 'ans_0': 0, 'ans_1': 3337, 'ans_2': 1105, 'ans_3': 267, 'ans_over_3': 2}, 'squad_unique_titles': {'comp': 0, 'ans_0': 0, 'ans_1': 8132, 'ans_2': 0, 'ans_3': 0, 'ans_over_3': 0, 'ans_2_refine': {'tot': 0, 'nf': 0, 'got_1': 0, 'got_2': 0, 'got_2_anseqtitle': 0, 'got_2_shortestlcstitle': 0, 'nf_shortestlcstitle': 0}}, 'hotpotqa_unique_titles': {'comp': 1278, 'ans_0': 0, 'ans_1': 3437, 'ans_2': 1274, 'ans_3': 0, 'ans_over_3': 0, 'ans_2_refine': {'tot': 1274, 'nf': 41, 'got_1': 1076, 'got_2': 157, 'got_2_anseqtitle': 80, 'got_2_shortestlcstitle': 66, 'nf_shortestlcstitle': 32}}}
cd_train, nf_train, nf_mdr_train = add_sequencing(beer_train, mdr_train, mdr_train_q_idx, titledict, docs) # Results: {'squad': {'comp': 0, 'ans_0': 0, 'ans_1': 58411, 'ans_2': 874, 'ans_3': 0, 'ans_over_3': 0}, 'hotpotqa': {'comp': 15162, 'ans_0': 0, 'ans_1': 37394, 'ans_2': 17675, 'ans_3': 4447, 'ans_over_3': 80}, 'squad_unique_titles': {'comp': 0, 'ans_0': 0, 'ans_1': 59285, 'ans_2': 0, 'ans_3': 0, 'ans_over_3': 0, 'ans_2_refine': {'tot': 0, 'nf': 0, 'got_1': 0, 'got_2': 0, 'got_2_anseqtitle': 0, 'got_2_shortestlcstitle': 0, 'nf_shortestlcstitle': 0}}, 'hotpotqa_unique_titles': {'comp': 15162, 'ans_0': 0, 'ans_1': 38745, 'ans_2': 20851, 'ans_3': 0, 'ans_over_3': 0, 'ans_2_refine': {'tot': 20851, 'nf': 510, 'got_1': 17515, 'got_2': 2826, 'got_2_anseqtitle': 1460, 'got_2_shortestlcstitle': 1166, 'nf_shortestlcstitle': 422}}}

#TODO Load docs into ES
# Note: Considered Updating hyperlinks with para_idx - as separate key - decided not to as don't have para id for test samples so no point

def beer_to_docs_map(beer_splits):
    """ Map beerqa train/dev samples to docs wiki paras in more convenient form
    """
    beer_map = {}
    for beer_split in beer_splits:
        for i, sample in enumerate(beer_split['data']):
            for m in sample['map']:
                d_idx = m['d_idx']  # Note d_idx is docs idx not wiki id
                p_idx = m['p_idx']
                if beer_map.get(d_idx) is None:
                    beer_map[d_idx] = {'hotpotqa': set(), 'squad':set()}
                beer_map[d_idx][sample['src']].add(p_idx)
    return beer_map

beer_map = beer_to_docs_map(beer_splits=[beer_dev, beer_train])  # 94489 paras mapped

def finalise_docs(docs, beer_map, index_name):
    """ docs preprocessing into ES format """
    final_docs = []
    for i, doc in enumerate(docs):
        for j, para in enumerate(doc['paras']):  # make paras 0 based and contiguous
            newid = doc['id'] + '_' + str(j)
            for_hp = False
            for_squad = False
            if beer_map.get(i) is not None:
                if j in beer_map[i]['hotpotqa']:
                    for_hp = True
                if j in beer_map[i]['squad']:
                    for_squad = True
            para['hpqa'] = for_hp
            para['squad'] = for_squad
            final_docs.append( {"_index": index_name, "_type": UES.TYPE, "_id": newid,
                                "_source": {"para_id": newid, "para_idx": j, "doc_id": doc['id'],
                                            "title": doc['title'], "title_unescaped": unescape(doc['title']),
                                            "text": para['text'], "for_hotpot": for_hp, "for_squad": for_squad }
                               } )
        if i % 500000 == 0:
            print(f"Processed: {i}")
    return final_docs

final_docs = finalise_docs(docs, beer_map, ES_INDEX)  # 35706771

client = UES.get_esclient()
print(UES.ping_client(client))
UES.create_index(client, UES.settings, index_name=ES_INDEX, force_reindex=True)
UES.index_by_chunk(client, final_docs, chunksize=500)


#TODO Add adversarial negatives


def get_hyperlinked_docs(docs, titledict, curr_d_idx, curr_p_idx, exclude=set()):
    """ Return indices of docs that are linked to by curr_d_idx, curr_p_idx
    """
    docs_linked_to_idx = set()
    docs_linked_to_id = set()
    for title in docs[curr_d_idx]['paras'][curr_p_idx]['hyperlinks_cased'].keys():
        new_title, status, d_idx = map_title_case(title, titledict)
        if d_idx != -1 and d_idx not in exclude:
            docs_linked_to_idx.add(d_idx)
            docs_linked_to_id.add(docs[d_idx]['id'])
    return docs_linked_to_idx, docs_linked_to_id


def get_paras(docs, d_idxs, p_idxs=[0]):
    """ Return paras for a set of doc idxs
    """
    paras = []
    for d_idx in d_idxs:
        para = ''
        for p_idx in p_idxs:
            if p_idx < len(docs[d_idx]['paras']) and p_idx > -1:
                para += ' ' + docs[d_idx]['paras'][p_idx]['text']
        paras.append( {'title': docs[d_idx]['title'], 'title_unescaped': unescape(docs[d_idx]['title']),
                       'text': para.strip()} )
    return paras


def get_paras_es(client, index_name, d_ids, p_idxs=[0]):
    """ retrieve paras from ES by exact match on para_id eg '1234_0'
    """
    paras = []
    for d_id in d_ids:        
        for p_idx in p_idxs:
            para_id = d_id + '_' + str(p_idx)
            hits = UES.exec_query(client, index_name, dsl=UES.term_query('para_id', para_id))
            paras += hits
    return paras                    


def get_adv_paras(client, docs, titledict, beer_sample, index_name):
    """ Return adversarial paras for a given sample through different methods.
    """
    adv_template = {'hlink':[], 'p':[], 'q_p':[]}  # {'type': [{'doc_id': [ {},..]}, ... ] }
    adversarial_paras = {} # {'title': adv_template}
    if beer_sample['src'] == 'hotpotqa':
        filter_dict = {"term": {"for_hotpot": False}}
    else:
        filter_dict = {"term": {"for_squad": False}}

    q = beer_sample['question']
    exclude_docs = {m['d_idx'] for m in beer_sample['map']} # exclude the gold para docs.
    for m in beer_sample['map']:
        curr_d_idx = m['d_idx']
        curr_p_idx = m['p_idx']
        title = docs[curr_d_idx]['title']
        if adversarial_paras.get(title) is None:
            adversarial_paras[title] = copy.deepcopy(adv_template)
        docs_linked_to_idx, docs_linked_to_id = get_hyperlinked_docs(docs, titledict, curr_d_idx=curr_d_idx, curr_p_idx=curr_p_idx, exclude=exclude_docs)
        hlink_paras_es = get_paras_es(client, index_name, d_ids=docs_linked_to_id, p_idxs=[0]) # take 1st hyperlinked para only
        adversarial_paras[title]['hlink'] += hlink_paras_es
    q_only_paras = UES.search(client, index_name, q, n_rerank=0, n_retrieval=5, filter_dic=filter_dict)    
    adversarial_paras['q_only'] = q_only_paras
    for title in beer_sample['para_agg']:
        para = title + ' ' + ' '.join(beer_sample['para_agg'][title])
        p_paras = UES.search(client, index_name, para, n_rerank=0, n_retrieval=5, filter_dic=filter_dict)
        adversarial_paras[title]['p'] = p_paras
        q_p_paras = UES.search(client, index_name, q + ' ' + para, n_rerank=0, n_retrieval=5, filter_dic=filter_dict)
        adversarial_paras[title]['q_p'] = q_p_paras
    return adversarial_paras
    

def add_adversaral_candidates(client, docs, titledict, index_name, beer_split):
    """ Add raw adversarial candidates to beer_split['neg_candidates']
    """
    print("Adding Adversarial candidates to a split....")
    for i, beer_sample in enumerate(beer_split['data']):
        adversarial_paras = get_adv_paras(client, docs, titledict, beer_sample, index_name)
        beer_sample['neg_candidates'] = adversarial_paras
        if i % 5000 == 0:
            print(f"Procesed: {i}")
    print("Finished!")        
    return


add_adversaral_candidates(client, docs, titledict, ES_INDEX, beer_dev)
add_adversaral_candidates(client, docs, titledict, ES_INDEX, beer_train)


beer_sample = beer_dev['data'][1]
adv_test = get_adv_paras(client, docs, titledict, beer_sample, ES_INDEX, verbose=True)

tst = UES.search(client, ES_INDEX, "Did Aristotle use a laptop?", n_rerank=0, n_retrieval=5)
tst = UES.search(client, ES_INDEX, "Aristotle", n_rerank=0, n_retrieval=5)

q = beer_dev['data'][0]['question']  # 'Which genus contains more species, Ortegocactus or Eschscholzia?'
paras = [p[0] + ' ' + ' '.join(p[1]) for p in beer_dev['data'][0]['para_agg'].items()]

curr_d_idx = beer_dev['data'][0]['map'][0]['d_idx']
exclude_docs = {m['d_idx'] for m in beer_dev['data'][0]['map']} # exclude the gold para docs
docs_linked_to_idx, docs_linked_to_id = get_hyperlinked_docs(docs, titledict, curr_d_idx=curr_d_idx, curr_p_idx=0, exclude=exclude_docs)
tst_hlink_paras = get_paras(docs, d_idxs=docs_linked_to_idx, p_idxs=[0])
tst_hlink_paras_es = get_paras_es(client, ES_INDEX, d_ids=docs_linked_to_id, p_idxs=[0])


mdr_negs = beer_dev['data'][0]['neg_paras']


tst = UES.search(client, ES_INDEX, q, n_rerank=0, n_retrieval=5)
tst = UES.search(client, ES_INDEX, q, n_rerank=0, n_retrieval=5, filter_dic={"term": {"for_hotpot": False}}) #works
tst = UES.search(client, ES_INDEX, q, n_rerank=0, n_retrieval=5, must_not={"term": {"for_hotpot": True}}) #works
tst = UES.search(client, ES_INDEX, q, n_rerank=0, n_retrieval=5, filter_dic=[{"term": {"para_idx": 0}}, {"term": {"for_hotpot": False}}]) # works
tst = UES.search(client, ES_INDEX, q, n_rerank=0, n_retrieval=5, filter_dic={ "range": { "para_idx": { "gte": 2, "lte": 8 }}}) # works. see https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-range-query.html

tst = UES.search(client, ES_INDEX, paras[0], n_rerank=0, n_retrieval=5, filter_dic={"term": {"for_hotpot": False}})
tst = UES.search(client, ES_INDEX, q + ' ' + paras[0], n_rerank=0, n_retrieval=5, filter_dic={"term": {"for_hotpot": False}})


tst = UES.search(client, ES_INDEX, '494525_25', fields=['para_id', 'doc_id'], n_rerank=0, n_retrieval=5) # works
tst = UES.search(client, ES_INDEX, 'Eschscholzia californica', fields=['title'], n_rerank=0, n_retrieval=5) # works

tst = UES.exec_query(client, ES_INDEX, dsl=UES.term_query('para_id', '494525_25')) # works


#TODO Write out final train/dev/test files for MDR
#TODO Write out corpus for MDR
#TODO Write out AISO corpus with para idxs added to hyperlinks
#TODO Write out AISO train/dev/test files



# LCS tests

# "neither references the other"
tst = [b for b in beer_dev['data'] if len(b['para_ans_titles'])==2 and len(b['para_ans_titles_refined'])==0]
get_overlap(tst[0]['question'], tst[0]['context'][0][1]+' '+tst[0]['context'][1][1], tst[0]['context'][0][0]) # (9, ['november', 'american', 'politician', 'served', 'south', 'carolina', 'general', 'assembly', '2007'])
get_overlap(tst[0]['question'], tst[0]['context'][2][1]+' '+tst[0]['context'][3][1], tst[0]['context'][2][0]) # (6, ['november', 'spratt', 'american', 'politician', 'served', 'carolina'])
# actually wrong in this case - Mulvaney should be first but the hyperlink from Spratt to Mulvaney is in a para not in the new context...

q = tst[1]['question']  # 'Minneapolis hip hop collective member that released album in 2006?'
c = tst[1]['context']
a = tst[1]['answers'] # ['P.O.S']
"""
[['Doomtree',
  'Doomtree is an American hip hop collective and record label based in Minneapolis, Minnesota. The collective has seven members: Dessa, Cecil Otter, P.O.S, Sims, Mike Mictlan, Paper Tiger, and Lazerbeak. The collective is known for incorporating a wide range of musical influences into their work with lyrical complexity and wordplay, and their annual "Doomtree Blowout" events held in Minneapolis venues to showcase their group performances and the Twin Cities music scene.'],
 ['Audition (album)',
  'Audition is the second solo studio album by American rapper P.O.S. It was released on Rhymesayers Entertainment in 2006. It peaked at number 45 on the "Billboard" Independent Albums chart.']]
"""
get_overlap(q, c[0][1], c[0][0])  # (3, ['minneapolis', 'hop', 'collective'])
get_overlap(q, c[1][1], c[1][0])  # (2, ['released', '2006'])
# correct here - p0 mentions member P.O.S -> p1 discusses P.O.S 's album release date

q = tst[2]['question']  # 'The USS Tortuga was named after the Dry Tortugas, a group of desert coral islets 60 miles west of which city with the motto "One Human Family"?'
c = tst[2]['context']
get_overlap(q, c[0][1]+' '+c[1][1], c[0][0])  # (2, ['the', 'west'])
get_overlap(q, c[2][1], c[2][0])  # (11, ['uss', 'tortuga', 'named', 'dry', 'tortugas', 'group', 'desert', 'coral', 'islets', '60', 'west'])
# correct here p1 actually has all the info necessary to answer the question as "Key West". link to P0 is actually unnecessary

q = tst[3]['question']  # 'The 2015 CrossFit Games were held at what multiple-use sports complex that is approximately 14 miles south of Downtown Los Angeles?'
c = tst[3]['context']
get_overlap(q, c[0][1], c[0][0])  # (9, ['the', 'sports', 'complex', 'approximately', '14', 'south', 'downtown', 'los', 'angeles'])
get_overlap(q, c[1][1], c[1][0])  # (5, ['the', '2015', 'crossfit', 'games', 'held'])
# correct here - actually either para id sufficent 


#TODO - test 2 "2 titles have ans embedded and both seem to reference each other"
tst1 = [b for b in beer_dev['data'] if len(b['para_ans_titles'])==2 and len(b['para_ans_titles_refined'])==2]

q = tst1[0]['question']  # 'In the "Star Trek" franchise, DeForest Kelley portrayed a character aboard which starship? '
c = tst1[0]['context']  # both titles do indeed reference each other Kirk = 0, McCooy = 1,2
a = tst1[0]['answers']   # ['USS "Enterprise"']
[m['bridge'] for m in mdr_dev if m['question'] == q] # ['James T. Kirk']
get_overlap(q, c[0][1], c[0][0])   #(7, ['``', 'star', 'trek', "''", 'franchise', 'portrayed', 'starship'])
get_overlap(q, c[1][1] + ' ' + c[2][1], c[1][0]) # (7, ['in', '``', 'trek', "''", 'deforest', 'kelley', 'character'])
# tie! if chose Kirk first, "could" hop to McCoy. McCoy para actually has all info necessary to answer.
# title heuristic would work here...

q = tst1[1]['question']  # 'Which American BMX rider hosted a show that spun off from "Real World" and "Road Rules"?'
c = tst1[1]['context']  # both titles do indeed reference each other Challenge = 0, Lavin = 1
a = tst1[1]['answers']   # ['T. J. Lavin']
[m['bridge'] for m in mdr_dev if m['question'] == q] # ['T. J. Lavin']
get_overlap(q, c[0][1], c[0][0])   # (10,  ['hosted', 'spun', '``', 'real', 'world', "''", '``', 'road', 'rules', "''"])
get_overlap(q, c[1][1] , c[1][0]) # (5, ['american', 'bmx', 'rider', '``', "''"])
# correct
# title heuristic would work here..

q = tst1[2]['question']  # 'What type of industry does Tony Bill and Flyboys have in common?'
c = tst1[2]['context']  # both titles do indeed reference each other 
a = tst1[2]['answers']   # ['film']
[m['bridge'] for m in mdr_dev if m['question'] == q] # ['Tony Bill']
get_overlap(q, c[0][1], c[0][0])   # (2, ['tony', 'bill'])
get_overlap(q, c[1][1] , c[1][0]) # (2, ['tony', 'bill'])
# tie! 

q = tst1[3]['question']  # 'After David Stern retired from being commissioner of the NBA, this american lawyer and businessman succeed him and is now the current commissioner who is he?'
c = tst1[3]['context']  # both titles do indeed reference each other Silver = 0,1 Commissioner = 2,3
a = tst1[3]['answers']   # ['Adam Silver']
[m['bridge'] for m in mdr_dev if m['question'] == q] # ['Adam Silver']
get_overlap(q, c[0][1] + ' ' + c[1][1], c[0][0])   # (5, ['david', 'stern', 'retired', 'commissioner', 'nba'])
get_overlap(q, c[2][1] + ' ' + c[3][1] , c[2][0]) # (4, ['david', 'stern', 'nba', 'commissioner'])
# opposite of mdr but it appears either para could give answer


#test "2 titles have ans embedded but only one references the other" - OK
tst1 = [b for b in beer_dev['data'] if len(b['para_ans_titles'])==2 and len(b['para_ans_titles_refined'])==1]

q = tst1[0]['question']  # 'Who is older Danny Green or James Worthy?'
c = tst1[0]['context']
a = tst1[0]['answers']  # ['James Worthy']  tst1[0]['para_ans_titles_refined'] = mdr_dev

q = tst1[1]['question'] # 'What word or phrase is found in both the history of Belgium and cockfighting?'
c = tst1[1]['context']
a = tst1[1]['answers']  # ['cockpit']  ['para_ans_titles_refined'] = mdr_dev
[m['bridge'] for m in mdr_dev if m['question'] == 'What word or phrase is found in both the history of Belgium and cockfighting?']

q = tst1[2]['question'] # 'What apartment complex originally constructed in 1927 is included in Manhattan Community Board 6?'
c = tst1[2]['context']
a = tst1[2]['answers']  # ['Tudor City']
[m['bridge'] for m in mdr_dev if m['question'] == 'What apartment complex originally constructed in 1927 is included in Manhattan Community Board 6?']
tst1[2]['para_ans_titles_refined']  # {'Tudor City'} = m['bridge']


#test "1 title has ans embedded"- OK
tst1 = [b for b in beer_dev['data'] if len(b['para_ans_titles'])==1]

q = tst1[2]['question'] # 'When did the car depicted on the cover of Pentastar: In the Style of Demons cease production?'
c = tst1[2]['context']
a = tst1[2]['answers']  # ['1974']
[m['bridge'] for m in mdr_dev if m['question'] == q] # ['Plymouth Barracuda']
tst1[2]['para_ans_titles']  # {'Plymouth Barracuda'}

q = tst1[3]['question'] # 'Horace Brindley played for what professional association football club that is based in the seaside town of Blackpool, Lancashire, England?'
c = tst1[3]['context']
a = tst1[3]['answers']  # 
[m['bridge'] for m in mdr_dev if m['question'] == q ] # ['Blackpool F.C.']
tst1[3]['para_ans_titles']  # {'Blackpool F.C.'}


### MDR dev/train exploration


def count_hpqa(split):
    """ Count bridge vs comparison stats
    """
    count_dict = {'tot': 0, 'br': 0, 'comp': 0, 'yn': 0, 'extractans': 0, 'br_yn': 0, 'comp_yn': 0}
    for i, sample in enumerate(split):
        count_dict['tot'] += 1
        if type(sample['answers']) != list:
            print(f'ERROR: {i} : answer not a list')
        ans = sample['answers'][0].lower().strip()
        if ans in ['yes', 'no']:
            count_dict['yn'] += 1
            if sample['type'] == 'comparison':
                count_dict['comp'] += 1
                count_dict['comp_yn'] += 1
            else:
                count_dict['br'] += 1
                count_dict['br_yn'] += 1
        else:
            count_dict['extractans'] += 1
            if sample['type'] == 'comparison':
                count_dict['comp'] += 1
            else:
                count_dict['br'] += 1
    print(f"Results: {count_dict}")
    return count_dict

# No bridge questions are yn:
count_hpqa(mdr_dev)     # Results: {'tot': 7405, 'br': 5918, 'comp': 1487, 'yn': 458, 'extractans': 6947, 'br_yn': 0, 'comp_yn': 458}
count_hpqa(mdr_train)   # Results: {'tot': 90447, 'br': 72991, 'comp': 17456, 'yn': 5481, 'extractans': 84966, 'br_yn': 0, 'comp_yn': 5481}


    



#### AISO dev/train exploration


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




