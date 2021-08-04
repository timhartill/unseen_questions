#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 12:59:58 2021

@author: tim hartill

Masking Objective Tests


"""
import numpy as np
import copy
import string

from transformers import AutoTokenizer, AutoModelForPreTraining

import spacy
nlp = spacy.load("en_core_web_sm")

m = "facebook/bart-large"
tokenizer = AutoTokenizer.from_pretrained(m)

t5 = AutoTokenizer.from_pretrained('t5-base')

qasc_corpus = []
with open("/data/thar011/data/unifiedqa/qasc_facts_selfsvised/dev.txt", "r") as f:
                for line in f:
                    qasc_corpus.append(tokenizer.bos_token + ' ' + line)
                    
wiki_corpus = []
with open("/data/thar011/data/unifiedqa/enwiki-20200511_selfsvised/dev.txt", "r") as f:
                for line in f:
                    wiki_corpus.append(tokenizer.bos_token + ' ' + line)



txt1 = tokenizer.bos_token + " " + "the rain is 123.25% heavier in Spain than NZ."
txt2 = txt1 + " the antimatter component has largess in banana land."




# this part occurs in preprocessing ie requires access to either the original string or the tokenised input 

def get_word_starts(toks, specialchar = 'Ġ', bos_token='<s>'):
    """ Get the beginning of each word in a list of tokenised text
        Return list of word beginning indices into toks
    """
    word_starts = [i for (i,t) in enumerate(toks) if t[0]==specialchar or t[0] in string.punctuation]
    if toks[0] == bos_token: # don't want to mask a bos token 
        word_starts.pop(0)   
    if word_starts[0] != 1:  # first non bos token is always a word start
        word_starts = [1] + word_starts
    return word_starts

# this part occurs on-the-fly during training in the dataset object

def get_spans(tok_idxs, toks_to_mask=0.11, avg_span_len=2, sd=0.75):
    """ Calculate number and length of spans for given input seq length
    """
    num_toks = len(tok_idxs)
    num_spans = int( (num_toks * toks_to_mask) / avg_span_len) + 1
    span_lengths = np.random.normal(avg_span_len, scale=sd, size=num_spans).round().astype('int')
    span_lengths = np.clip(span_lengths, 1, avg_span_len+4)    
    return span_lengths


def merge_intervals(in_list):
    """ Merge overlapping intervals in a list
    """
    in_list.sort(key=lambda interval: interval[0])
    merged = [in_list[0]]
    for current in in_list:
        previous = merged[-1]
        if current[0] <= previous[1]:
            previous[1] = max(previous[1], current[1])
        else:
            merged.append(current)
    return merged


def wwsc_select_spans(tok_idxs, span_lengths, word_starts, verbose = False):
    """ Convert a set of span lengths into actual start/end token positions
    """
    num_toks = len(tok_idxs)
    num_words = len(word_starts)
    replace_spans = []
    for length in span_lengths:
        if verbose: print(f'Processing length: {length}')
        span_start_idx = np.random.choice(num_words)
        span_start = word_starts[span_start_idx]
        if verbose: print(f'Start: {span_start}  tok: {tok_idxs[span_start]}  length:{length}')
        if span_start + length > num_toks:
            length = num_toks - span_start
            if verbose: print(f'Length past eos, truncating length to {length}')
        else:
            for next_wordstart in word_starts[span_start_idx+1:]:
                if verbose: print(f"Finding word boundary. Checking {next_wordstart} {tok_idxs[next_wordstart]}")
                if next_wordstart >= span_start+length:
                    length = next_wordstart - span_start
                    if verbose: print(f"Found! New length: {length}  last token in span: {tok_idxs[span_start+length]}")
                    break
        span_end = span_start + length
        replace_spans.append( [span_start, span_end]  )
    replace_spans = merge_intervals(replace_spans)  # aggregate overlaps
    return replace_spans


def mask_words(tok_idxs, replace_spans, mask_seq):
    """ Given a list of token indices,  + an array of spans [[start1, end1], [start2, end2], ...]
        + a list of mask substitutions
        return a masked version of toks plus the list of masked spans 
    """
    replaced_toks = []
    tmp_tok_idxs = tok_idxs.copy()
    ctr = 0
    for replace_span in replace_spans:
        replaced_toks.append( mask_seq[ctr] + tok_idxs[replace_span[0]:replace_span[1]] )
        ctr += 1
        if ctr > 18:  # use mask_seq[19] to as answer end indicator
            ctr = 0
        first = True
        for i in range(replace_span[0], replace_span[1]):
            if first:
                tmp_tok_idxs[i] = -8888
                first = False
            else:
                tmp_tok_idxs[i] = -9999
    new_tok_idxs = []
    ctr = 0
    for tok in tmp_tok_idxs:
        if tok != -9999:
            if tok == -8888:
                new_tok_idxs.extend(mask_seq[ctr])
                ctr += 1
                if ctr > 18:  # use mask_seq[19] to as answer end indicator
                    ctr = 0
            else:    
                new_tok_idxs.append(tok)
    return new_tok_idxs, replaced_toks
    


#TODO might have to do .lower after tokenizing in order to allow spacy better opportunity to find named entities...

txt3 = tokenizer.bos_token + "the rain is 123.25% heavier in Spain than NZ."
txt4 = "the rain is 123.25% heavier in Spain than NZ."


toktxt = tokenizer.tokenize(txt1)

toktxt = tokenizer.tokenize(txt2)

toktxt = tokenizer.tokenize(txt3)
toktxt = tokenizer.tokenize(txt4)


word_starts = get_word_starts(toktxt)  # in reality add 1 to word starts for the extra bos token thats added later in manual_batch_encode

print(f"toktxt: {toktxt}")
print(f"word_starts: {word_starts}")
tok_idxs = tokenizer.convert_tokens_to_ids(toktxt)
print(f"tok_idxs: {tok_idxs}")

span_lengths = get_spans(tok_idxs, toks_to_mask=0.11, avg_span_len=2)  #toks_to_mask is not the literal % that will be masked since we adjust upwards to word boundaries
print(f"span_lengths: {span_lengths}")
replace_spans = wwsc_select_spans(tok_idxs, span_lengths, word_starts)
print(f"replace_spans: {replace_spans}")
new_tok_idxs, replaced_toks, ratio = mask_words(tok_idxs, replace_spans, mask_token=tokenizer.mask_token_id, verbose=True)
print(f"New toks: {tokenizer.decode(new_tok_idxs)}")
print(f"Masked: {replaced_toks}  {[tokenizer.decode(r) for r in replaced_toks]}")

ratios = []
num_spans = []
new_idxs = []
replaced = []
for line in qasc_corpus:
    toktxt = tokenizer.tokenize(line)
    word_starts = get_word_starts(toktxt)
    tok_idxs = tokenizer.convert_tokens_to_ids(toktxt)
    span_lengths = get_spans(tok_idxs, toks_to_mask=0.11, avg_span_len=2)  #toks_to_mask is not the literal % that will be masked since we adjust upwards to word boundaries
    replace_spans = wwsc_select_spans(tok_idxs, span_lengths, word_starts)
    new_tok_idxs, replaced_toks, ratio = mask_words(tok_idxs, replace_spans, mask_token=tokenizer.mask_token_id)
    ratios.append(ratio)
    num_spans.append(len(replaced_toks))
    new_idxs.append(new_tok_idxs)
    replaced.append(replaced_toks)
ratios = np.array(ratios)
num_spans = np.array(num_spans)
print(f"Number: {ratios.shape[0]}  Mean masked toks ratio: {ratios.mean():.2f}  Max masked toks:{ratios.max():.2f}  Min masked toks:{ratios.min():.2f}")
print(f"Mean spans: {num_spans.mean():.2f}  Max spans:{num_spans.max():.2f}  Min spans:{num_spans.min():.2f}")


ratios = []
num_spans = []
new_idxs = []
replaced = []
for line in wiki_corpus:
    toktxt = tokenizer.tokenize(line)
    word_starts = get_word_starts(toktxt)
    tok_idxs = tokenizer.convert_tokens_to_ids(toktxt)
    span_lengths = get_spans(tok_idxs, toks_to_mask=0.11, avg_span_len=2)  #toks_to_mask is not the literal % that will be masked since we adjust upwards to word boundaries
    replace_spans = wwsc_select_spans(tok_idxs, span_lengths, word_starts)
    new_tok_idxs, replaced_toks, ratio = mask_words(tok_idxs, replace_spans, mask_token=tokenizer.mask_token_id)
    ratios.append(ratio)
    num_spans.append(len(replaced_toks))
    new_idxs.append(new_tok_idxs)
    replaced.append(replaced_toks)
ratios = np.array(ratios)
num_spans = np.array(num_spans)
print(f"Number: {ratios.shape[0]}  Mean masked toks ratio: {ratios.mean():.2f}  Max masked toks:{ratios.max():.2f}  Min masked toks:{ratios.min():.2f}")
print(f"Mean spans: {num_spans.mean():.2f}  Max spans:{num_spans.max():.2f}  Min spans:{num_spans.min():.2f}")

print(wiki_corpus[0])
print(tokenizer.decode(new_idxs[0]))
print(replaced[0])

print(wiki_corpus[10])
print(tokenizer.decode(new_idxs[10]))
print(replaced[10])


# SSM tests
txt=tokenizer.bos_token + " " + "john smith is a nice person. john smith from apple is looking at buying u.k. startup for $1 billion in July 2020 or perhaps 1/6/23 or failing that 2024\nHello world.\nApples are good fruit to eat\nAre new zealand fruit or australian vegetables better for you? Astronomers look for the bright stars that orbit dark partners in the same way. The North Star can be used to find your way if you're lost in the dark. The north star can be used to find your way if you're lost in the dark"


def ner(instr, verbose=False):
    """ Perform named entity recognition on text and return a list of named entities, numbers, dates etc
    """
    ner_list = []
    doc = nlp(instr.replace('\\n \n', ''))    
    for ent in doc.ents:
        if verbose: print(ent.text, '"' + ent.text_with_ws + '"', ent.start_char, ent.end_char, ent.label_)
        ner_list.append(ent.text_with_ws)
        #ner_list.append( {'txt_with_ws': ent.text_with_ws, 'start':ent.start_char, 'end': ent.end_char, 'type': ent.label_} )
    return ner_list


def find_tok_idx(toks, start, end):
    """ Convert start/end indices in original text to token indices in tokenised text list
    Unused, did map_ners approach instead..
    """
    tok_start = -1
    tok_end = -1
    curr_str_idx_start = 0
    for i, tok in enumerate(toks):
        curr_str_idx_end = curr_str_idx_start + len(tok.replace('Â','').replace('Ä','').replace('Å',''))
        if start >= curr_str_idx_start and start <= curr_str_idx_end:
            tok_start = i
        if end >= curr_str_idx_start and end <= curr_str_idx_end:
            tok_end = i+1
            break
        curr_str_idx_start = curr_str_idx_end
    return tok_start, tok_end


def find_sub_list(sublst1, sublst2, lst):
    """ Return start/end indices of all occurences of sublist in list
        Note: Can't tell whether the tokens match with/out a preceding space so must try both ways
    """
    results=[]
    sll=len(sublst1)
    for ind in (i for i,e in enumerate(lst) if e==sublst1[0]):
        if lst[ind:ind+sll]==sublst1:
            results.append((ind,ind+sll))
    sll=len(sublst2)
    for ind in (i for i,e in enumerate(lst) if e==sublst2[0]):
        if lst[ind:ind+sll]==sublst2:
            results.append((ind,ind+sll))  
    results = list(set(results))
    new_results = []
    for l in results:
        new_results.append( list(l) )
    return new_results  


def map_ners(toks, ners, tokenizer, verbose = False):
    """ Map list of NERs previously identified on raw text to token ids
    """
    unique_ner = list(set([ w.strip(string.punctuation+' ').strip() for w in ners ]))
    tok_map = []
    final_ner = []
    for n in unique_ner:
        ner_txt_tok = tokenizer.tokenize(n)
        ner_txt_tok2 = tokenizer.tokenize(' ' + n)
        found_list = find_sub_list(ner_txt_tok, ner_txt_tok2, toks)
        if verbose: 
            print(f"Orig: {n}") 
            for tok_start, tok_end in found_list:
                print(f"tokens: {toks[tok_start:tok_end]}")
            if len(found_list) == 0:
                print(f"NOT FOUND: {n} 1:{ner_txt_tok} 2:{ner_txt_tok2}")
        if len(found_list) > 0:
            tok_map.append(found_list)
            final_ner.append(n)
    return final_ner, tok_map


ners = []
for line in qasc_corpus:
    ners.append( ner( line ) )  #spacy thinks land.\\n and \\n in general is a person..
num_ners = [len(n) for n in ners]
num_ners_np = np.array(num_ners)
print(f" Num: {len(num_ners)}  Mean:{num_ners_np.mean():.2f}  Max:{num_ners_np.max():.2f}  Min:{num_ners_np.min():.2f}")


wiki_ners = []
for line in wiki_corpus:
    wiki_ners.append( ner( line ) )  #spacy thinks land.\\n and \\n in general is a person..
num_ners = [len(n) for n in wiki_ners]
num_ners_np = np.array(num_ners)
print(f" Num: {len(num_ners)}  Mean:{num_ners_np.mean():.2f}  Max:{num_ners_np.max():.2f}  Min:{num_ners_np.min():.2f}")


toks = tokenizer.tokenize(wiki_corpus[1])
item_ners, item_ners_ids = map_ners(toks, wiki_ners[1], tokenizer)  # [[[22, 30],[1,9]], [[8, 15]], [[61, 67]]]

tsttoks = toks[:29]
numtoks = len(tsttoks)
new_ners_ids = []
new_ners = []
print(f"Starting item_ners: {item_ners}  Starting item_ners_ids:{item_ners_ids}")
for i, item_ids in enumerate(item_ners_ids):
    print(i, item_ids)
    new_item_id_list = []
    for j, id in enumerate(item_ids):
        print(i, j, id)
        if id[1] <= numtoks: 
            new_item_id_list.append(id)
        else:
            print(f'Delete {id} since {id[1]} > {numtoks}')
    if new_item_id_list:
        new_ners.append( item_ners[i] )
        new_ners_ids.append(new_item_id_list)            
print(f"Ending item_ners: {new_ners}  Ending item_ners_ids:{new_ners_ids}")



def test_ner(c, ner, i, verbose = False):
    wiki0 = tokenizer.tokenize(c[i].lower())
    #if verbose: print(c[i].lower())
    #if verbose: print(wiki0)
    unique_ner = set([ w['txt_with_ws'].lower().strip(string.punctuation+' ').strip() for w in ner[i] ])
    for n in unique_ner:
        ner_txt_tok = tokenizer.tokenize(n)
        ner_txt_tok2 = tokenizer.tokenize(' ' + n)
        #if verbose: print(f"{ner_txt} tokenised: 1:{ner_txt_tok}  2: {ner_txt_tok2}")
        found_list = find_sub_list(ner_txt_tok, ner_txt_tok2, wiki0)
        if verbose: 
            print(f"Orig: {n}")  #  Start:{n['start']}  End:{n['end']}")
            #print(f"Tok idx found: {found_list}")
            for tok_start, tok_end in found_list:
                print(f"tokens: {wiki0[tok_start:tok_end]}")
        if len(found_list) == 0:
            print(f"{i}NOT FOUND: {n} 1:{ner_txt_tok} 2:{ner_txt_tok2}")

test_ner(wiki_corpus, wiki_ners, 2, verbose=True)

for i in range(len(wiki_corpus)):
    test_ner(wiki_corpus, wiki_ners, i)

for i in range(len(qasc_corpus)):
    test_ner(qasc_corpus, ners, i)


wiki0 = tokenizer.tokenize("Bibekananda Agarwala " + wiki_corpus[1]+" Bibekananda Agarwala.")
print(wiki_corpus[1])
print(wiki0)
wiki0rejoined = ''.join(wiki0)
print(wiki0rejoined)
print(f"len wiki0: {len(wiki_corpus[1])} len wiki0rejoined: {len(wiki0rejoined)}")
for n in ners[1]:
    ner_txt = n['txt_with_ws'].strip()
    ner_txt_tok = tokenizer.tokenize(ner_txt)
    ner_txt_tok2 = tokenizer.tokenize(' ' + ner_txt)
    print(f"{ner_txt} tokenised: 1:{ner_txt_tok}  2: {ner_txt_tok2}")
    found_list = find_sub_list(ner_txt_tok, ner_txt_tok2, wiki0)
    print(f"Orig: {n['txt_with_ws']}  Start:{n['start']}  End:{n['end']}")
    print(f"Tok idx found: {found_list}")
    #print(f"wiki_corpus: #{wiki_corpus[1][n['start']:n['end']]}#")
    #print(f"wiki0rejoined: #{wiki0rejoined.replace('Â','').replace('Ä','')[n['start']:n['end']]}#")  ## Not exactly identical § tokenizes to Â§ so strip Â
    #tok_start, tok_end = find_tok_idx(wiki0, n['start'], n['end'])
    #print(f"tok start: {tok_start}  tok end: {tok_end}")
    for tok_start, tok_end in found_list:
        print(f"tokens: {wiki0[tok_start:tok_end]}")


args.append_another_bos=True
args.strip_single_quotes=False
args.do_lowercase=True
args.indiv_digits=False

questions = wiki_corpus
questions.extend(wiki_corpus)
metadata = [(0, len(wiki_corpus)),(len(wiki_corpus), len(wiki_corpus)+len(wiki_corpus))]

question_input = manual_batch_encode(questions, 
                                     tokenizer,
                                     None,
                                     args,
                                     [True, False],
                                     metadata,
                                     truncation=True,
                                     pad=False,
                                     max_length=512)

answers = [''] * len(wiki_corpus)
answer_input = manual_batch_encode(answers, 
                                     tokenizer,
                                     None,
                                     args,
                                     [True],
                                     metadata,
                                     truncation=True,
                                     pad=False,
                                     max_length=100)


decoder_input_ids, decoder_attention_mask = answer_input["input_ids"], answer_input["attention_mask"]


print(question_input.keys())
word_starts = question_input["word_starts"]  
ners_ids = question_input["ners_ids"]
input_ids, attention_mask = question_input["input_ids"], question_input["attention_mask"]

def disp(i):
    print('ORIG:', wiki_corpus[i])
    print('DECODED:', tokenizer.decode(input_ids[i]))
    print('INPUTIDS:', input_ids[i], len(input_ids[i]))
    #print(attention_mask[i], len(attention_mask[i]))
    print('NERS', ners_ids[i])
    print('WS', word_starts[i])
    print("LASTWORDTOKS:", input_ids[i][word_starts[i][-1]:], 'LASTWORD:"' + tokenizer.decode(input_ids[i][word_starts[i][-1]:]) + '"')
    print("FIRSTNER", input_ids[i][ners_ids[i][0][0][0]: ners_ids[i][0][0][1]])
    print('FIRSTNERDEC#'+tokenizer.decode(input_ids[i][ners_ids[i][0][0][0]: ners_ids[i][0][0][1]]) + '"')
    print('LASTNERDEC#'+tokenizer.decode(input_ids[i][ners_ids[i][-1][0][0]: ners_ids[i][-1][0][1]]) + '#')

disp(0)
disp(1)
disp(2)
disp(3)
lens = [len(i) for i in wiki_corpus]
m = max(lens)
midx = [j for j,l in enumerate(lens) if l == m]  # 64

disp(64)


from torch.utils.data import Dataset

class dl(Dataset):
    def __init__(self, thedata):
        self.dldata = thedata
        
    def __len__(self):
        return 10
    
    def __getitem__(self, idx):
        self.dldata[idx] = -9999
        

class t1(object):
    def __init__(self):
        self.data = [1,2,3,4,5,6,7,8,9,10]
        self.mydl = None
    
    def load(self):
        self.mydl = dl(self.data)    

tst = t1()
tst.load()
tst.data  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
tst.mydl.dldata[0] = -99
tst.data    #[-99, 2, 3, 4, 5, 6, 7, 8, 9, 10]
tst.mydl.__getitem__(2)
tst.data   #[-99, 2, -9999, 4, 5, 6, 7, 8, 9, 10]


# tests - run args etc in cli.py + imports from run.py first...
#logger = None
tokenizer = AutoTokenizer.from_pretrained(args.model) 

args.output_dir = '/data/thar011/out/unifiedqa_bart_large_TEST'
args.do_lowercase = True
args.append_another_bos = True
args.verbose=True
# Must set up logger after setting up args.output_dir ...
log_filename = "{}log.txt".format("" if args.do_train else "eval_")

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                datefmt='%m/%d/%Y %H:%M:%S',
                level=logging.INFO,
                handlers=[logging.FileHandler(os.path.join(args.output_dir, log_filename)),
                          logging.StreamHandler()])
logger = logging.getLogger(__name__)
logger.info(args)
logger.info(args.output_dir)
logger.info("MLM TESTS...")

args.ssm_prob = 0.5
args.wwsc_toks_to_mask = 0.11
args.wwsc_avg_span_len=2
args.wwsc_span_len_sd=0.75
args.add_mask_char='_' # 'NONE' works
args.add_mask_ctr=True

# test unified_data.py:
args.is_unifiedqa = True
args.mixture = 'unifiedqa,synthetic_textual,synthetic_numeric'
args.mixture = 'unifiedqa'
args.mixture = 'arc_hard,strategy_qa_facts_selfsvised,strategy_qa,qasc_facts_selfsvised'
args.train_file = '/data/thar011/data/unifiedqa/train.tsv'
args.predict_file = '/data/thar011/data/unifiedqa/dev.tsv'
dev_data = UnifiedQAData(logger, args, args.predict_file, False)
dev_data = UnifiedQAData(logger, args, args.train_file, True)

print(dev_data.data.keys())     # dict_keys(['arc_hard', 'strategy_qa_facts_selfsvised', 'strategy_qa', 'qasc_facts_selfsvised'])
print(len(dev_data.data['arc_hard']['question']))  #299   train: 1119
print(len(dev_data.data['strategy_qa_facts_selfsvised']['question']))  #849  train: 8402
print(len(dev_data.data['strategy_qa']['question']))  #229  train: 2061
print(len(dev_data.data['qasc_facts_selfsvised']['question']))  #2304  train: 19438

logger.info(args)


#tokenizer: ['<s>', '</s>', '<unk>', '<pad>', '<mask>']  [0, 2, 3, 1, 50264]
dev_data.load_dataset(tokenizer)
print(dev_data.dataset.is_training)
dev_data.dataset.bos_token_id  # 0
dev_data.dataset.eos_token_id  # 2
dev_data.dataset.mask_token_id # 50264
dev_data.dataset.pad_token_id  # 1
print(dev_data.dataset.no_question_label)  # [2362, 11445] = 'no mask'
print(dev_data.dataset.mask_seq) # [[50264, 1215, 288], [50264, 1215, 134], [50264, 1215, 176], [50264, 1215, 246], [50264, 1215, 306], [50264, 1215, 245], [50264, 1215, 401], [50264, 1215, 406], [50264, 1215, 398], [50264, 1215, 466], [50264, 1215, 698], [50264, 1215, 1225], [50264, 1215, 1092], [50264, 1215, 1558], [50264, 1215, 1570], [50264, 1215, 996], [50264, 1215, 1549], [50264, 1215, 1360], [50264, 1215, 1366], [50264, 1215, 1646]]
print(dev_data.dataset.selfsupervised)  # [False, True, False, True]
print(dev_data.dataset.metadata)        # [(0, 299), (299, 1148), (1148, 1377), (1377, 3681)] training: [[0, 1119], [1119, 9521], [9521, 11582], [11582, 31020]]


#for uqa: add/test build_objective_indx, get_parentdata_indx
print(dev_data.dataset.objective[0]) #False
print(dev_data.dataset.objective[299]) #True
print(dev_data.dataset.objective[299+849]) #False
print(dev_data.dataset.objective[299+849+229]) #True
print(dev_data.dataset.unified_dataset)     # ['arc_hard', 'strategy_qa_facts_selfsvised', 'strategy_qa', 'qasc_facts_selfsvised']

get_parentdata_indx(1119, dev_data.dataset.metadata, dev_data.dataset.unified_dataset)
get_parentdata_indx(1119-1, dev_data.dataset.metadata, dev_data.dataset.unified_dataset)
get_parentdata_indx(1119+8402, dev_data.dataset.metadata, dev_data.dataset.unified_dataset)
get_parentdata_indx(1119+8402+2061, dev_data.dataset.metadata, dev_data.dataset.unified_dataset)
get_parentdata_indx(1119+8402+2061+19438-1, dev_data.dataset.metadata, dev_data.dataset.unified_dataset)  # ('qasc_facts_selfsvised', 2303)

print(dev_data.data['arc_hard']['question'][0])
print(dev_data.data['arc_hard']['answer'][0])
print(dev_data.data['qasc_facts_selfsvised']['question'][0])
print(dev_data.data['qasc_facts_selfsvised']['answer'][0])  #''

idx = 0 #1119
ds, ds_idx = get_parentdata_indx(idx, dev_data.dataset.metadata, dev_data.dataset.unified_dataset)
dev_data.dataset.parent_data[ds].keys()     # dict_keys(['id', 'question', 'answer'])


print(dev_data.dataset.parent_data[ds]['id'][ds_idx])
dev_data.dataset.parent_data[ds]['id'][ds_idx] += 'TEST'
print(dev_data.data[ds]['id'][ds_idx])  # writes correctly
print(dev_data.data[ds]['answer'][ds_idx])

print(dev_data.dataset.word_starts[idx])
print(dev_data.dataset.ners_ids[idx])
print(dev_data.dataset.attention_mask[idx]) # 1s for no ssupervised, empty for ssupervised
print(dev_data.dataset.decoder_attention_mask[idx])  # ditto
print(dev_data.dataset.input_ids[idx])        # two 0,s start, 2 at end, no padding
print(dev_data.dataset.decoder_input_ids[idx]) # ditto but [] for ssvise

# if not is_training returns (input_ids, attention_mask) else returns (input_ids, attention_mask, decoder_input_ids, decoder_attention_mask)
print(dev_data.dataset.__len__())

sample = dev_data.dataset.__getitem__(idx)
print(len(sample))  # not is_training: 2 or 4 for is_training
print(sample[0], sample[0].shape)  # input_ids tensor padded [512], 2 bos, eos
print(sample[1], sample[1].shape)  # input attn mask tensor padded [512], 1s, 0's

print(tokenizer.decode(sample[0]))
print(dev_data.data[ds]['answer'][ds_idx])

print(sample[2], sample[2].shape)  # decoder_input_ids tensor padded [100], 2 bos, eos
print(sample[3], sample[3].shape)  # decoder attn mask tensor padded [100], 1s, 0's

print(tokenizer.decode(sample[2]))





# test data.py:
trainset = 'strategy_qa_facts_selfsvised'
testset = 'strategy_qa_facts_selfsvised'
args.is_unifiedqa = False
args.train_file = '/data/thar011/data/unifiedqa/'+trainset+'/train.tsv'
args.predict_file = '/data/thar011/data/unifiedqa/'+testset+'/dev.tsv'
dev_data = QAData(logger, args, args.predict_file, False)
dev_data = QAData(logger, args, args.train_file, True)  # 


print(dev_data.data[0])

logger.info(args)


#tokenizer: ['<s>', '</s>', '<unk>', '<pad>', '<mask>']  [0, 2, 3, 1, 50264]
dev_data.load_dataset(tokenizer)
print(dev_data.dataset.is_training)
dev_data.dataset.bos_token_id  # 0
dev_data.dataset.eos_token_id  # 2
dev_data.dataset.mask_token_id # 50264
dev_data.dataset.pad_token_id  # 1
print(dev_data.dataset.no_question_label)  # [2362, 11445] = 'no mask'
print(dev_data.dataset.mask_seq) # [[50264, 1215, 288], [50264, 1215, 134], [50264, 1215, 176], [50264, 1215, 246], [50264, 1215, 306], [50264, 1215, 245], [50264, 1215, 401], [50264, 1215, 406], [50264, 1215, 398], [50264, 1215, 466], [50264, 1215, 698], [50264, 1215, 1225], [50264, 1215, 1092], [50264, 1215, 1558], [50264, 1215, 1570], [50264, 1215, 996], [50264, 1215, 1549], [50264, 1215, 1360], [50264, 1215, 1366], [50264, 1215, 1646]]
print(dev_data.dataset.selfsupervised)  # [False, True, False, True]
print(dev_data.dataset.metadata)        # [(0, 299), (299, 1148), (1148, 1377), (1377, 3681)]
#for uqa: add/test build_objective_indx, get_parentdata_indx


idx = 0

print(dev_data.dataset.parent_data[idx])
dev_data.dataset.parent_data[idx]['id'] += 'TEST'
print(dev_data.data[idx])  # writes correctly

print(dev_data.dataset.word_starts[idx])
print(dev_data.dataset.ners_ids[idx])
print(dev_data.dataset.attention_mask[idx]) # 1s for no ssupervised, empty for ssupervised
print(dev_data.dataset.decoder_attention_mask[idx])  # ditto
print(dev_data.dataset.input_ids[idx])        # two 0,s start, 2 at end, no padding
print(dev_data.dataset.decoder_input_ids[idx]) # ditto but [] for ssvise

# if not is_training returns (input_ids, attention_mask) else returns (input_ids, attention_mask, decoder_input_ids, decoder_attention_mask)
print(dev_data.dataset.__len__())

sample = dev_data.dataset.__getitem__(idx)
print(len(sample))  # not is_training: 2 or 4 for is_training
print(sample[0], sample[0].shape)  # input_ids tensor padded [512], 2 bos, eos
print(sample[1], sample[1].shape)  # input attn mask tensor padded [512], 1s, 0's

print(tokenizer.decode(sample[0]))
print(dev_data.data[idx])

print(sample[2], sample[2].shape)  # decoder_input_ids tensor padded [100], 2 bos, eos
print(sample[3], sample[3].shape)  # decoder attn mask tensor padded [100], 1s, 0's

print(tokenizer.decode(sample[2]))



fname = '/data/thar011/data/unifiedqa/train-v2--uncased-xbos-BartTokenizedFast-_unifiedqa_strategy_qa_strategy_qa_facts_selfsvised.json'




