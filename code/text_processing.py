#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 14:22:47 2021

@author: tim hartill 

Text processing functions

Code adapted from elsewhere is noted in comments by respective functions.

"""

import random
import numpy as np

import string
import re

from nltk import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
STOPWORDS = set(nltk_stopwords.words('english'))

from w2n import word_to_num   # from https://github.com/ag1988/injecting_numeracy/blob/master/pre_training/gen_bert/create_examples_n_features.py

import spacy
from spacy.matcher import Matcher
from spacy.util import filter_spans
from spacy.tokens import Token
Token.set_extension('lefts', default=[], force=True)
Token.set_extension('rights', default=[], force=True)
Token.set_extension('relative_position', default=0, force=True)

nlp = spacy.load("en_core_web_sm")

# For verb phrase matching:
pattern = [{'POS': 'VERB', 'OP': '?'},
           {'POS': 'ADV', 'OP': '*'},
           {'POS': 'AUX', 'OP': '*'},
           {'POS': 'VERB', 'OP': '+'}]

# instantiate a Matcher instance
matcher = Matcher(nlp.vocab)
matcher.add("Verb phrase", [pattern])


# from https://github.com/facebookresearch/UnsupervisedQA/blob/master/unsupervisedqa/configs.py (Lewis et al)
NOUNPHRASE_LABEL = 'NOUNPHRASE'
CLOZE_MASKS = {
    'PERSON': 'IDENTITYMASK',
    'NORP': 'IDENTITYMASK',
    'FAC': 'PLACEMASK',
    'ORG': 'IDENTITYMASK',
    'GPE': 'PLACEMASK',
    'LOC': 'PLACEMASK',
    'PRODUCT': 'THINGMASK',
    'EVENT': 'THINGMASK',
    'WORKOFART': 'THINGMASK',
    'WORK_OF_ART': 'THINGMASK',
    'LAW': 'THINGMASK',
    'LANGUAGE': 'THINGMASK',
    'DATE': 'TEMPORALMASK',
    'TIME': 'TEMPORALMASK',
    'PERCENT': 'NUMERICMASK',
    'MONEY': 'NUMERICMASK',
    'QUANTITY': 'NUMERICMASK',
    'ORDINAL': 'NUMERICMASK',
    'CARDINAL': 'NUMERICMASK',
    NOUNPHRASE_LABEL: 'NOUNPHRASEMASK'
}

HEURISTIC_CLOZE_TYPE_QUESTION_MAP = {
    'PERSON': ['Who', ],
    'NORP': ['Who', ],
    'FAC': ['Where', ],
    'ORG': ['Who', ],
    'GPE': ['Where', ],
    'LOC': ['Where', ],
    'PRODUCT': ['What', ],
    'EVENT': ['What', ],
    'WORKOFART': ['What', ],
    'WORK_OF_ART': ['What', ],
    'LAW': ['What', ],
    'LANGUAGE': ['What', ],
    'DATE': ['When', ],
    'TIME': ['When', ],
    'PERCENT': ['How much', 'How many'],
    'MONEY': ['How much', 'How many'],
    'QUANTITY': ['How much', 'How many'],
    'ORDINAL': ['How much', 'How many'],
    'CARDINAL': ['How much', 'How many'],
    NOUNPHRASE_LABEL: ['What', 'What', ''],  #TJH Added
}


def white_space_fix(text):
    """ Remove repeated white space in a string"""
    return ' '.join(text.split())


def replace_chars(instr): 
    """ Very basic text preprocessing """
    outstr = instr.replace("’", "'").replace("‘", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!").replace(" ,", ",")
    return outstr.replace('“', '"').replace('”','"').replace("\t", " ").replace("\n", "")


def format_sentence(sentence, capitalize=True, add_endchar=True, endchar='.', strip=[]):
    """ Format sentence with basic preprocessing then optionally by capitalizing first letter 
        and putting 'endchar' on end if there isn't one.
        Optionally strip any strings in [strip] from sentence
    """
    if strip is not None and strip != []:
        if type(strip) == str:
            strip = [strip]
        for s in strip:
            sentence = sentence.replace(s, '')        
    sentence = replace_chars(white_space_fix(sentence))
    if sentence != '':
        if add_endchar and sentence[-1] not in ['.', '?', '!']:
            sentence += endchar
        if capitalize:
            sentence = sentence[0].upper() + sentence[1:]            
    return sentence


# Adapted from https://github.com/Neutralzz/RefQA (Li et al)
def identity_translate(cloze_question, mask_type=''):
    """ Replace mask with WH word..."""
    if mask_type != '':
        return cloze_question.replace(mask_type, random.choice(HEURISTIC_CLOZE_TYPE_QUESTION_MAP[mask_type]))
    for mask in HEURISTIC_CLOZE_TYPE_QUESTION_MAP:
        if mask in cloze_question:
            return cloze_question.replace(mask, random.choice(HEURISTIC_CLOZE_TYPE_QUESTION_MAP[mask]))
    raise Exception('\'{}\' should have one specific masked tag.'.format(cloze_question))
        

def word_shuffle(tokens, word_shuffle_param):
    length = len(tokens)
    noise = np.random.uniform(0, word_shuffle_param, size=(length ) )
    word_idx = np.array([1.0*i for i in range(length)])

    scores = word_idx + noise
    scores += 1e-6 * np.arange(length)
    permutation = scores.argsort()
    new_s = [ tokens[idx] for idx in permutation ]
    return new_s


def word_dropout(tokens, word_dropout_param):
    length = len(tokens)
    if word_dropout_param == 0:
        return tokens
    assert 0 < word_dropout_param < 1

    keep = np.random.rand(length) >= word_dropout_param
    #if length:
    #    keep[0] = 1
    new_s =  [ w for j, w in enumerate(tokens) if keep[j] ]
    return new_s


def word_mask(tokens, word_mask_param, mask_str='<mask>'):
    length = len(tokens)
    if word_mask_param == 0:
        return tokens
    assert 0 < word_mask_param < 1

    keep = np.random.rand(length) >= word_mask_param
    new_s =  [ w if keep[j] else mask_str  for j, w in enumerate(tokens)]
    return new_s


def noisy_clozes_translate(cloze_question, params=[2, 0.2, 0.1], mask_type=''):
    wh = None
    if mask_type != '':
        cloze_question = cloze_question.replace(mask_type,'')
        wh = random.choice(HEURISTIC_CLOZE_TYPE_QUESTION_MAP[mask_type])
    else:    
        for mask in HEURISTIC_CLOZE_TYPE_QUESTION_MAP:
            if mask in cloze_question:
                cloze_question = cloze_question.replace(mask,'')
                wh = random.choice(HEURISTIC_CLOZE_TYPE_QUESTION_MAP[mask])
                break
    tokens = word_tokenize(cloze_question)
    tokens = word_shuffle(tokens, params[0])
    tokens = word_dropout(tokens, params[1])
    tokens = word_mask(tokens, params[2])
    return wh+' '+(' '.join(tokens))


def parsing_tree_dfs(node):
    N = len(node._.lefts) + len(node._.rights)
    if N == 0:
        return node.text

    text = ''
    for child in node._.lefts:
        text += parsing_tree_dfs(child)+' '
    text += node.text
    for child in node._.rights:
        text += ' '+parsing_tree_dfs(child)
    return text


def reform_tree(node):
    if node.text in HEURISTIC_CLOZE_TYPE_QUESTION_MAP.keys():
        node._.lefts = []
        return True
    flag = False
    res = None
    for child in node._.lefts:
        flag |= reform_tree(child)
        if flag:
            node._.lefts.remove(child)
            node._.lefts = [child] + node._.lefts
            break
    if not flag:
        for child in node._.rights:
            flag |= reform_tree(child)
            if flag:
                node._.rights.remove(child)
                node._.lefts = [child] + node._.lefts
                break
    return flag

def reformulate_question(question, parser, reform_version=1, mask_type=''):    
    doc = parser(question)
    roots = []
    for token in doc:
        token._.lefts = [child for child in token.lefts]
        token._.rights = [child for child in token.rights]
        if token.dep_ == 'ROOT':
            roots.append(token)
    ### reformulate ###
    for root in roots:
        if reform_version == 1:
            result = reform_tree(root)
        else:
            result = False
        if result:
            roots.remove(root)
            roots = [root] + roots
    ### tree to sequence ###
    new_question = ''
    for root in roots:
        new_question += ' ' + parsing_tree_dfs(root)  
        
    if mask_type != '':
        wh = random.choice(HEURISTIC_CLOZE_TYPE_QUESTION_MAP[mask_type])
        mask = mask_type
    else:    
        for mask in HEURISTIC_CLOZE_TYPE_QUESTION_MAP:
            if mask in question:
                wh = random.choice(HEURISTIC_CLOZE_TYPE_QUESTION_MAP[mask])
                break
    new_question = new_question.replace(mask, wh)
    return new_question.strip()


# Adapted from https://github.com/awslabs/unsupervised-qa (Fabbri et al)
def generate_template_awb(text, answer_str, sampled_ngram):
    """
    answer_str: eg 'NOUNPHRASE'
    sampled_ngram: 'WHAT'
    If cloze-style is “[FragmentA] [PERSON] [FragmentB]”, then:

    "[FragmentA], who [FragmentB]?" - AWB
    """

    # need to use \W+ to ensure word boundaries and not part of a word
    template = re.sub(
        r'\W+{}\W+'.format(re.escape(answer_str)),
        ', {} '.format(sampled_ngram),
        ' ' + text + ' ')

    # remove leading comma if the replacement was the first word
    template = re.sub(r'^,\s*', '', template)
    template = template.strip()
    if template.endswith(','):
        template = template[:len(template)-1]

    return template


def generate_template_wba(text, answer_str, sampled_ngram):
    """
    answer_str: eg 'NOUNPHRASE'
    sampled_ngram: 'WHAT'
    If cloze-style is “[FragmentA] [PERSON] [FragmentB]”, then:

    "Who [FragmentB] [FragmentA]?" - WBA
    """
    # need to use \W+ to ensure word boundaries and not part of a word
    template = re.sub(
        r'^(.*?)\W+{}\W+(.*?)\W*$'.format(re.escape(answer_str)),
        r'{} \2, \1'.format(sampled_ngram),
        ' ' + text + ' ')

    template = template.strip()
    # remove leading comma if the replacement was the first word
    template = re.sub(r'^,\s*', '', template)
    template = re.sub(r'\s+', ' ', template)  # regex above may have created double spaces
    return template


def generate_template(question, mask_type='', template='wba'):
    """ Front end on template generation
    """
    if mask_type != '':
        wh = random.choice(HEURISTIC_CLOZE_TYPE_QUESTION_MAP[mask_type])
        mask = mask_type
    else:    
        for mask in HEURISTIC_CLOZE_TYPE_QUESTION_MAP:
            if mask in question:
                wh = random.choice(HEURISTIC_CLOZE_TYPE_QUESTION_MAP[mask])
                break
    if template == 'wba':
        newq = generate_template_wba(question, mask, wh)
    else:
        newq = generate_template_awb(question, mask, wh)
    return newq


def make_into_question(text, method, mask_type=''):
    """ Wrapper for different question-making strategies
        Assumes text already has the answer replaced with a key from HEURISTIC_CLOZE_TYPE_QUESTION_MAP
        Depending upon dataset, mask_type could be set in which case it will process faster
    """
    if method == 'id':  # identity
        q = identity_translate(text, mask_type=mask_type)
    elif method == 'nc':  # noisy cloze
        q = noisy_clozes_translate(text, params=[2, 0.2, 0.0], mask_type=mask_type)
    elif method == 'rq':  # reformulate question using spacy
        q = reformulate_question(text, nlp, mask_type=mask_type, reform_version=1)
        if q in HEURISTIC_CLOZE_TYPE_QUESTION_MAP[NOUNPHRASE_LABEL]: # ['What', '']
            q = generate_template(text, mask_type=mask_type)  # occasionally question reformatting fails
    elif method in ['wba','awb']:  # Fabbri method
        q = generate_template(text, mask_type=mask_type, template=method)        
    else:
        print(f"ERROR: Unknown questionizing method: {method}")
        q = text
    q = q.strip()
    if len(q)==0 or q[-1] != '?':
        q += '?'   
    return q    
    

def extract_and_print(txt):
    """ Test fn illustrating extraction of verb phrases etc
        txt="john smith is a nice person. john smith from apple is looking at buying u.k. startup for $1 billion in July 2020 or perhaps 1/6/23 or failing that 2024\nHello world.\nApples are good fruit to eat\nAre new zealand fruit or australian vegetables better for you? Astronomers look for the bright stars that orbit dark partners in the same way. The North Star can be used to find your way if you're lost in the dark. The north star can be used to find your way if you're lost in the dark"
    """
    print('TEXT:', txt)
    doc = nlp(txt)
    for n in doc:  #POS
        print('POS', n.text, n.pos_, n.idx, n.shape_, n.text_with_ws)
    for n in doc.noun_chunks:  #Noun Chunks
        print('NOUN CHUNK',n.text, n.start_char, n.end_char, n.text_with_ws)
    for ent in doc.ents:  #NER
        print('NER', ent.text, '"' + ent.text_with_ws + '"', ent.start_char, ent.end_char, ent.label_, HEURISTIC_CLOZE_TYPE_QUESTION_MAP[ent.label_])
    matches = matcher(doc)  #Verb chunks
    spans = [doc[start:end] for _, start, end in matches]  # matches = [(id,start_char, end_char),..]
    print ('VERB CHUNK', filter_spans(spans))


# adapted from https://github.com/castorini/transformers-arithmetic/blob/main/main.py
def convert_to_10ebased(number: str, split_type: str=None, invert_number: bool=False) -> str:
    signal = None
    if number[0] == '-':
        signal = '-'
        number = number[1:]

    digitpos = number.find('.')
    if digitpos != -1:
        number = number.replace('.', '')
        i = (len(number) - digitpos) * -1
    else:
        i = 0
        
    output = []
    for digit in number[::-1]:
        if split_type is None:
            output.append('10e' + str(i))
        elif split_type == 'underscore':
            output.append('10e_' + str(i))
        elif split_type == 'character':
            output.append(' '.join('D' + str(i) + 'E'))
        else:
            raise Exception(f'Wrong split_type: {split_type}')
        output.append(digit)
        i += 1

    if signal:
        output.append(signal)

    # The output is already inverted. If we want it to _not_ be inverted, then we invert it.
    if not invert_number:
        output = output[::-1]
    return ' '.join(output)


#Adapted from https://github.com/ag1988/injecting_numeracy/blob/master/pre_training/gen_bert/create_examples_n_features.py    
def convert_word_to_number(word: str):
    """
    Returns number if convertable from word string otherwise None
    """
    # strip all punctuations from the sides of the word, except for the negative sign
    punctuations = string.punctuation.replace('-', '')
    if word[0] == '.':
        punctuations = punctuations.replace('.', '')
    word = word.strip(punctuations)
    # some words may contain the comma as deliminator
    word = word.replace(",", "")
    # word2num will convert hundred, thousand ... to number, but we skip it.
    if word in ["hundred", "thousand", "million", "billion", "trillion"]:
        return None
    try:
        number = word_to_num(word)
    except ValueError:
        try:
            number = int(word)
        except ValueError:
            try:
                number = float(word)
            except ValueError:
                number = None
    return number


def normalize_num(instr, norm=''):
    """ Normalise numbers found in input string and return normalised string
    """
    doc = nlp(instr)  #spacy tokenization
    newtext = []
    for token in doc:
        if token.pos_ == 'NUM' or (len(token.text)>1 and set(token.text).issubset(set('0123456789,-.'))):
            norm_word = convert_word_to_number(token.text)
            if norm_word is not None:
                norm_num = str(norm_word)
                if norm == '10e':
                    norm_num = convert_to_10ebased(norm_num)
                if token.text_with_ws[-1] == ' ':
                    norm_num += ' '
                newtext.append(norm_num)
            else:
                newtext.append(token.text_with_ws)        
        else:
            newtext.append(token.text_with_ws)
    outstr = ''.join(newtext)   
    return outstr    


# Adapted from BERT/wordpiece version split_digits from https://github.com/ag1988/injecting_numeracy/blob/master/pre_training/gen_bert/create_examples_n_features.py:
def split_digits_special(wps, special='Ġ'): #-> List[str]:
    """
    Further split numeric tokens accommodating arbitrary special char 
    For t5 special='▁'  (not underscore)
    For bart/roberta special='Ġ'
    eg tokenizer_bart.tokenize('124567890')-> ['12', '45', '678', '90']
      split_digits_special(tokenizer_bart.tokenize('124567890'), special='Ġ') -> ['1', '2', '4', '5', '6', '7', '8', '9', '0']
    eg2 split_digits_special(tokenizer_t5.tokenize('the rain in 124567890 99 999 is similar to a55.'), special='▁')
    """
    toks = []
    for wp in wps:
        if set(wp).issubset(set(special+'0123456789.-$,^')) and set(wp) != {special}: # numeric wp - split digits
            for i, dgt in enumerate(list(wp.replace(special, ''))):
                prefix = special if (wp.startswith(special) and i == 0) else ''
                toks.append(prefix + dgt)
        else:
            toks.append(wp)
    return toks


def verb_chunks(instr, add_dot=True, verbose=False):
    """ Extract verb chunks 
    """
    tmpstr = instr.rstrip()
    tmpstr = tmpstr.rstrip('\\n').rstrip()
    if add_dot:
        if tmpstr[-1] not in ['.', '?', '!']:
            tmpstr += '.'
    else:
        tmpstr += ' ' 
    doc = nlp(instr)
    matches = matcher(doc)  #Verb chunks
    spans = [doc[start:end] for _, start, end in matches]  # matches = [(id,start_char, end_char),..]
    out_spans = filter_spans(spans)
    out_spans = [v.text.strip() for v in out_spans]
    if verbose:
        print ('VERB CHUNK', out_spans)
    return out_spans


def ner(instr, add_nphrases = True, add_dot=True, verbose=False, return_types=False):
    """ Perform named entity recognition on text and return a list of named entities, numbers, dates etc
        Optionally also extract noun phrases
        Note: optionally add full stop if missing for slightly better results on the last word
    """
    ner_list = []
    ner_types = []
    just_ners = []
    tmpstr = instr.rstrip()
    tmpstr = tmpstr.rstrip('\\n').rstrip()
    if add_dot:
        if tmpstr[-1] not in ['.', '?', '!']:
            tmpstr += '.'
    else:
        tmpstr += ' ' 
    doc = nlp(tmpstr)    
    for ent in doc.ents:
        if verbose: print(ent.text, '"' + ent.text_with_ws + '"', ent.start_char, ent.end_char, ent.label_)
        ner_list.append(ent.text_with_ws.strip())
        ner_types.append(str(ent.label_))
        just_ners.append(ent.text.strip(string.punctuation+' '))
        #ner_list.append( {'txt_with_ws': ent.text_with_ws, 'start':ent.start_char, 'end': ent.end_char, 'type': ent.label_} )
    if add_nphrases:    
        for n in doc.noun_chunks:  #Noun Chunks
            if verbose: print('NOUN CHUNK',n.text_with_ws, n.start_char, n.end_char)
            if not [ne for ne in just_ners if ne.find(n.text.strip(string.punctuation+' ')) != -1]: # don't include noun phrases that are part of a named entity
                ner_list.append(n.text.strip(string.punctuation+' '))
                ner_types.append(NOUNPHRASE_LABEL)
    if return_types:
        return ner_list, ner_types
    return ner_list







