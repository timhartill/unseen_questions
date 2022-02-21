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
AISO_FILE = '/data/thar011/gitrepos/AISO/data/corpus/beer_v1.tsv'
BEER_WIKI_SAVE = '/home/thar011/data/beerqa/enwiki-20200801-pages-articles-compgen.json'
BEER_TITLE_SAVE = '/home/thar011/data/beerqa/enwiki-20200801-titledict-compgen.json'
BEER_DEV = '/home/thar011/data/beerqa/beerqa_dev_v1.0.json'
BEER_TRAIN = '/home/thar011/data/beerqa/beerqa_train_v1.0.json'

MDR_DEV = '/data/thar011/gitrepos/compgen_mdr/data/hotpot/hotpot_dev_with_neg_v0.json'
MDR_TRAIN = '/data/thar011/gitrepos/compgen_mdr/data/hotpot/hotpot_train_with_neg_v0.json'




tstfile = '/AA/wiki_00.bz2'

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
    """ Split doc into paras and process each para
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
            pidx = 0 # make paras 0 based and contiguous
            for j, para in enumerate(doc['paras']):
                if para['text'].strip() != '':
                    newid = doc['id'] + '_' + str(pidx)
                    pidx += 1
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


def add_sequencing(beer_split, mdr_split, mdr_split_q_idx):
    """ follow mdr paper and calculate paragraph sequencing for beerqa:
        if 'bridge': final "bridge" para is one mentioning the answer span. 
                     TODO if the answer span is in both, the one that has its title mentioned in the other passage is treated as the second.
                     TODO Use bqa dev/train doc/para mappings from above
    Merge question type and neg paras from mdr for HPQA
    Can simply concatenate paras with same title to make noisier samples or exclude paras+titles that don't contain the answer for cleaner samples
    Nb: All squad dev have exactly 1 para but 874 squad train have 2. 
            
    """
    nf_idx = []
    nf_mdr = []
    count_dict = {'squad': {'comp':0, 'ans_0': 0, 'ans_1': 0, 'ans_2': 0, 'ans_3':0, 'ans_over_3': 0},
                  'hotpotqa': {'comp':0, 'ans_0': 0, 'ans_1': 0, 'ans_2': 0, 'ans_3':0, 'ans_over_3': 0},
                  'squad_unique_titles': {'comp':0, 'ans_0': 0, 'ans_1': 0, 'ans_2': 0, 'ans_3':0, 'ans_over_3': 0},
                  'hotpotqa_unique_titles': {'comp':0, 'ans_0': 0, 'ans_1': 0, 'ans_2': 0, 'ans_3':0, 'ans_over_3': 0},
                 }
    for i,sample in enumerate(beer_split['data']):
        sample['type'] = ''
        sample['neg_paras'] = []
        if sample['src'] == 'hotpotqa':
            mdr_idx = mdr_split_q_idx.get(sample['question'].strip().lower())
            if mdr_idx is not None:
                sample['type'] = mdr_split[mdr_idx]['type']
                sample['neg_paras'] = mdr_split[mdr_idx]['neg_paras']
            else:
                nf_mdr.append(i)
        if type(sample['answers']) != list:
            print(f'ERROR: {i} : answer not a list')
        ans = sample['answers'][0].strip().lower()
        para_seq = []
        para_title = set()
        for j, (title, para) in enumerate(sample['context']):
            if sample['src'] == 'hotpotqa' and sample['type'] == 'comparison': #ans in ['yes', 'no']:
                continue
            if ans in para.lower() or ans in title.lower():  # and ans not in ['yes', 'no']:
                
                para_seq.append(j)
                para_title.add(title)
        sample['para_has_ans'] = para_seq
        sample['para_ans_titles'] = para_title
        if len(para_title) == 0:
            if sample['src'] == 'squad' or (sample['src'] == 'hotpotqa' and sample['type'] != 'comparison'): #ans not in ['yes', 'no']):
                nf_idx.append(i)   
        
        
    for i,sample in enumerate(beer_split['data']):
        ans = sample['answers'][0].strip().lower()
        l = len(sample['para_has_ans'])
        num_titles = len(sample['para_ans_titles'])
        
        if sample['src'] == 'hotpotqa' and sample['type'] == 'comparison': #ans in ['yes', 'no']:
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
            
        if sample['src'] == 'hotpotqa' and sample['type'] == 'comparison':  #ans in ['yes', 'no']:
            count_dict[sample['src']+'_unique_titles']['comp'] += 1
        elif num_titles == 0:
            count_dict[sample['src']+'_unique_titles']['ans_0'] += 1
        elif num_titles == 1:
            count_dict[sample['src']+'_unique_titles']['ans_1'] += 1
        elif num_titles == 2:
            count_dict[sample['src']+'_unique_titles']['ans_2'] += 1
        elif num_titles == 3:
            count_dict[sample['src']+'_unique_titles']['ans_3'] += 1
        else:
            count_dict[sample['src']+'_unique_titles']['ans_over_3'] += 1
            
    print(f"Results: {count_dict}")
    print(f"No answer span found: {len(nf_idx)}")
    print(f"Number of hpqa samples not matched to MDR: {len(nf_mdr)}")
    return count_dict, nf_idx, nf_mdr

cd_dev, nf_dev, nf_mdr_dev = add_sequencing(beer_dev, mdr_dev, mdr_dev_q_idx)  # Results: {'squad': {'comp': 0, 'ans_0': 0, 'ans_1': 7970, 'ans_2': 162, 'ans_3': 0, 'ans_over_3': 0}, 'hotpotqa': {'comp': 1278, 'ans_0': 0, 'ans_1': 3337, 'ans_2': 1105, 'ans_3': 267, 'ans_over_3': 2}, 'squad_unique_titles': {'comp': 0, 'ans_0': 0, 'ans_1': 8132, 'ans_2': 0, 'ans_3': 0, 'ans_over_3': 0}, 'hotpotqa_unique_titles': {'comp': 1278, 'ans_0': 0, 'ans_1': 3437, 'ans_2': 1274, 'ans_3': 0, 'ans_over_3': 0}}
cd_train, nf_train, nf_mdr_train = add_sequencing(beer_train, mdr_train, mdr_train_q_idx) # Results: {'squad': {'comp': 0, 'ans_0': 0, 'ans_1': 58411, 'ans_2': 874, 'ans_3': 0, 'ans_over_3': 0}, 'hotpotqa': {'comp': 15162, 'ans_0': 0, 'ans_1': 37394, 'ans_2': 17675, 'ans_3': 4447, 'ans_over_3': 80}, 'squad_unique_titles': {'comp': 0, 'ans_0': 0, 'ans_1': 59285, 'ans_2': 0, 'ans_3': 0, 'ans_over_3': 0}, 'hotpotqa_unique_titles': {'comp': 15162, 'ans_0': 0, 'ans_1': 38745, 'ans_2': 20851, 'ans_3': 0, 'ans_over_3': 0}}



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




