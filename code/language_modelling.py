#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 13:15:21 2021

@author: tim hartill

Routines for language modelling e.g. using GPT-x


"""
import os
import numpy as np
import copy

import utils
import eval_metrics
import text_processing

question_templates = {'who': "Who is {phrase}?",
                        'what': 'What is {phrase}?',
                        'general':'{phrase}',
                        'where': 'Where is {phrase}?',
                        'when': 'When did {phrase} exist?',
                        'size':'What size is {phrase}?',}

exclude_ner_types = set(['DATE','TIME','PERCENT','MONEY','QUANTITY','ORDINAL','CARDINAL'])

def load_prompt_template(infile):
    """ Load a prompt file """
    with open(infile, 'r') as fp: 
        template = fp.read()
    if template[-1] == '\n':
        template = template[:-1]  # strip final \n if exists
    return template


def load_templates(infiles):
    """ load a set of prompt templates """
    if type(infiles) != list:
        infiles = [infiles]
    template_list = []    
    for infile in infiles:
        template = load_prompt_template(infile)
        template_list.append(template)
    return template_list


def get_prompt_samples_and_eval_samples(train_samples, select_total=100, select_prompt=7, select_eval=30, seed=42):
    """ Select select_total random indices from a train set to form a set of indices 
        considered for either inclusion in a prompt (first select_prompt samples) 
        or inclusion in a "test set" of train examples (last select_eval samples)
        used to evaluate goodness of generated explanations.
    """
    num_q = len(train_samples)
    np.random.seed(seed)
    rand_indices = np.random.choice(num_q, select_total, replace=False)
    prompt_indices = rand_indices[:select_prompt]
    test_indices = rand_indices[select_total-select_eval:]
    return prompt_indices, test_indices, rand_indices


def fill_prompt_template(template, query=None, taskprompt=None, example_inputs=[], example_outputs=[],
                         saveas=None):
    """ Fill slots in a template and optionally save template to a prompt file called 'saveas'
    Will replace all:
        {question} or {QUESTION} with query
        {taskprompt} or {TASKPROMPT} with taskprompt
        {EXAMPLE}Input '{EXAMPLENUM}: {EXAMPLEINPUT}
        Knowledge: {EXAMPLEOUTPUT} with k examples. Assumes {EXAMPLEOUTPUT} is the end of the example template and this and {EXAMPLE} only occur once.
    """
    prompt = template
    if taskprompt is not None:
        prompt = prompt.replace('{taskprompt}', taskprompt, 1).replace('{TASKPROMPT}', taskprompt, 1)
    if query is not None:
        prompt = prompt.replace('{question}', query).replace('{QUESTION}', query)
    start_example = prompt.find('{EXAMPLE}')
    if start_example != -1:
        end_example = prompt.find('{EXAMPLEOUTPUT}')
        if end_example != -1:
            end_example += len('{EXAMPLEOUTPUT}')
            if prompt[end_example] == '\n':
                end_example += 1
            example_template = prompt[start_example+len('{EXAMPLE}'):end_example]
            new_examples = ''
            for i, (ex_in, ex_out) in enumerate(zip(example_inputs, example_outputs)):
                curr = example_template.replace('{EXAMPLEINPUT}', ex_in).replace('{EXAMPLEOUTPUT}', ex_out).replace('{EXAMPLENUM}', str(i+1))
                new_examples += curr
            prompt = prompt.replace('{EXAMPLE}' + example_template, new_examples)
            prompt = prompt.replace('{EXAMPLENUM}', str(i+2))
    prompt = prompt.lstrip()
    if saveas is not None:
        with open(saveas, 'w') as f:
            f.write(prompt)
    return prompt


def create_template(orig_template_file, new_template_file, ds_jsonl, prompt_indices, qkey = 'q_only', explkey='context'):
    """ create and save a new prompt template based on an existing template and some examples
    Set new_template_file to None to not save the new template as a file
    """
    orig_template = load_prompt_template(orig_template_file)
    example_inputs = utils.return_sublist(ds_jsonl, prompt_indices, key=qkey)
    example_outputs = utils.return_sublist(ds_jsonl, prompt_indices, key=explkey)
    new_template= fill_prompt_template(orig_template, 
                                      example_inputs=example_inputs,
                                      example_outputs=example_outputs,
                                      saveas=new_template_file)
    return new_template


def generate_continuations(templates, model, tokenizer, queries, example_inputs=[], example_outputs=[], verbose=False, lower=False,
                           max_input_length=512, **generator_args):
    """ Generate LM continuations for [templates] filled with [queries] and optionally with [example_inputs] & [example_outputs]

    Examples:
    #Nucleus/top p: (will not return sequences_scores if output_scores=True)
    qasc_completions = generate_continuations(qasc_2_templates, model, tokenizer, test_questions, verbose=True,
                                              example_inputs=example_inputs, example_outputs=example_outputs, max_input_length=1000, 
                                              do_sample=True, max_new_tokens=64, top_k=0, top_p=0.9, temperature=0.7,
                                              num_return_sequences=10, output_scores=False, return_dict_in_generate=True)    
    #Beam search: (will return sequences_scores if output_scores=True)
    qasc_completions = generate_continuations(qasc_2_templates, model, tokenizer, test_questions, verbose=True,
                                              example_inputs=example_inputs, example_outputs=example_outputs, max_input_length=1000, 
                                              num_beams=4, early_stopping=True, min_length=1, max_new_tokens=64,
                                              num_return_sequences=1, output_scores=False, return_dict_in_generate=True)
    #top k: (will not return sequences_scores if output_scores=True)
    qasc_completions = generate_continuations(qasc_2_templates, model, tokenizer, test_questions, verbose=True,
                                              example_inputs=example_inputs, example_outputs=example_outputs, max_input_length=1000, 
                                              do_sample=True, max_new_tokens=64, top_k=50, temperature=0.7,
                                              num_return_sequences=10, output_scores=False, return_dict_in_generate=True)  
        
    Generally, [templates] previously loaded with load_templates() above.
    Returns [ { 'template_idx': 'raw': ['output 0', 'output 1', ..., 'output 9'] } ] where each row idx corresponds to query idx
    """
    if verbose:
        print(f"Generating continuations for max in len:{max_input_length} Generator params: {generator_args}")
    if type(templates) != list:
        templates = [templates]
    if type(queries) != list:
        queries = [queries]
    outlist = []
    num_q = len(queries)
    for j, query in enumerate(queries):
        if verbose:
            print(f"Processing {len(templates)} templates for query {j+1} of {num_q}..")
        out = {}
        for i, template in enumerate(templates):
            prompt = fill_prompt_template(template, query=query, 
                                            example_inputs=example_inputs, example_outputs=example_outputs)

            res = utils.run_model(prompt, model, tokenizer, indiv_digits=False, norm_numbers=False, 
                            max_input_length=max_input_length, verbose=verbose,
                            lower=lower, append_eos=False, prepend_bos=False, only_decode_new=True, cut_at_nl=True,
                            **generator_args)
            out[str(i)] = {}
            out[str(i)]['raw'] = res.preds.copy()
            if verbose and j==0 and i==0:
                print(f"RESULTS KEYS: {res.keys()}")
        outlist.append(out)
    return outlist


def preds_basic_filter(preds, min_length=4, remove_strings=[]):
    """ Given a list of predicted explanations, perform basic filtering
    - remove with length < min_length
    - remove duplicates
    - remove substring sample where it is a substring of another
    - remove normalised predictions starting with any of ['strings'] (sometimes GPTJ generates predictions like "Input 9: some irrelevant thing")
    """
    preds = [p.strip() for p in preds if len(p.strip()) >= min_length]
    preds_norm = [eval_metrics.normalize_answer(p) for p in preds]
    delete_idx = []
    for i, pred in enumerate(preds_norm):
        for j, pred_comp in enumerate(preds_norm):
            if i != j:
                if pred in pred_comp and j not in delete_idx:
                    delete_idx.append(i)
                    break
    for rem in remove_strings:
        for i, pred in enumerate(preds_norm):
            if pred.startswith(rem.lower()):
                delete_idx.append(i)
    #print('delete:', delete_idx)
    pred_tuple = []
    for i, pred in enumerate(preds):
        if i in delete_idx:
            pred_tuple.append( (pred, True) )
        else:
            pred_tuple.append( (pred, False) )
    preds = [pt[0] for pt in pred_tuple if pt[1] == False]
    return preds


def calc_measure_basic(preds, compareto, measure_fn=None):
    """ Return a basic relevance measure of each item in list preds to compareto """
    if measure_fn is None:
        measure_fn = eval_metrics.get_f1
    scores = []
    for pred in preds:
        score = measure_fn(pred, compareto)    
        scores.append(score)
    return scores


def filter_continuations(sample_list, min_length=4, remove_strings=['input']):
    """ Apply various heuristic fns to refine continuations to relevant subset
    sample_list format (after running add_key()): 
        [ {'question':'full question with context in UQA format', 
           'answer':'ans', 'q_only':'Question only?', 
           'mc_options': 'mc options, 'context':'non-mc context if any', 
           'expls':{'0':{'raw':['expl 1', 'expl 2', ...]} } }]
    Adds keys to 'expls': eg 
    """
    for i, s in enumerate(sample_list):
        #q = s['q_only']
        for k in s['expls']:  # basic filtering on individual prompt template preds '0', '1', ..
            e = s['expls'][k]
            e['filtered'] = preds_basic_filter(e['raw'], min_length=min_length, remove_strings=remove_strings)
            #e['f1_to_q'] = calc_measure_basic(e['filtered'], compareto=q, measure_fn=eval_metrics.get_f1)
    return
    

def make_soft_unique(preds, stop_when_lessthan=50, N=2, smooth=True, norm=True, verbose=False):
    """ Return a selection of predictions that are maximally diverse by progressively eliminating preds with high overlap to other preds
    preds: list of strings
    """
    scorer = eval_metrics.SelfBleu(N, smooth)
    newpreds = preds.copy()
    selfbleu = scorer.compute_metric(newpreds, norm=norm)
    if verbose:
        print(f"Selfbleu:{selfbleu} Preds remaining:{len(newpreds)}")
    while selfbleu > stop_when_lessthan and len(newpreds) > 2:
        most_similar_idx = np.argmax(scorer.self_bleus)
        newpreds.pop(most_similar_idx)
        selfbleu = scorer.compute_metric(newpreds, norm=norm)
        if verbose:
            print(f"Selfbleu:{selfbleu} Preds remaining:{len(newpreds)}")
    return newpreds        
            

def gen_expl(templates, model, tokenizer, queries, example_inputs=[], example_outputs=[], verbose=False, lower=False,
             max_input_length=1000, gen_depth=1, filter_min_len=4, filter_remove_strings=['input'], su_stage_1_stop=-1, 
             add_noun=['general', 'where', 'when', 'size'], add_verb=['general'], outfile=None,
             **generator_args):
    """ Iteratively generate explanation using a set of general templates. 
    templates: list of generic template(s) to apply
    add_noun = list of question_templates keys to apply to nouns using first template in templates only for depth 0 only
    add_verb = list of question_templates keys to apply to verbs using first template in templates only for depth 0 only
    (A) Generate num_return_sequences candidate explanation components for "queries". Combine these, do basic filtering and make "slightly" soft-unique unless su_stage_1_stop=-1
    (B) Then for each remaining explanation component generate num_return_sequences explanation components per (A). Repeat gen_depth times.
    Returns: outlist list of {'question':q, 'expl_depth':[ ['depth 0 expl components 1', 'd0 ec 2', ..], ['depth 1 expl components', ..], .. ],
                              'noun':{'general':{'Aristotle':['Sentence 1', 'Sentence 2', ...], 'a laptop':[...]}, ...}
                              'verb':{'general':{'running':[...], 'jumping':[...]}}
                             }
    """
    num_t = len(queries)
    outlist = [{'question':q, 'expl_depth':[], 'noun':{}, 'verb':{}} for q in queries]  #expl_depth = list of [expl components] generated at each depth
    num_partial = 0
    initial_mode = 'w'
    if outfile is not None:
        if os.path.exists(outfile):
            partial_outfile = utils.load_jsonl(outfile)
            num_partial = len(partial_outfile)
            assert num_partial <= num_t, f"ERROR: query count {num_t} is less than partial count {num_partial} from {outfile}. Exiting!"
            if num_partial > 0:
                initial_mode = 'a'
            for t,p in enumerate(partial_outfile):
                if p['question'] != outlist[t]['question']:
                    assert num_partial <= num_t, f"ERROR: File question {t} doesnt match input question: File:{p['question']} Input:{outlist[t]['question']}. File name:{outfile}. Exiting!"
                outlist[t] = p
                
    for t in range(num_partial, num_t):
        if verbose:
            print(f"Processing Q:{t} {outlist[t]['question']}")
        test_questions = outlist[t]['question']
        if add_noun != []:
            ners, ner_types = text_processing.ner(test_questions, return_types=True)
            ners_filtered = []
            for i, ntype in enumerate(ner_types):
                if ntype not in exclude_ner_types:
                    ners_filtered.append(ners[i])
            ners_filtered = preds_basic_filter(ners_filtered, min_length=0, remove_strings=[]) # filter substrings in favour of longest noun phrases
            curr_out = {}  # {'general':{'Aristotle':[...], 'a laptop':[...]}, ...}
            for tem in question_templates.keys():
                if tem in add_noun:                        
                    curr_out[tem]={}
                    for ner in ners_filtered:
                        question = question_templates[tem].replace('{phrase}', ner)
                        if verbose:
                            print(f"Q:{t} QT:{tem} NP: {ner} Question:{question}...")
                        curr_completions = generate_continuations(templates[0], model, tokenizer, 
                                                                  question, verbose=verbose, lower=lower,
                                                                  example_inputs=example_inputs, example_outputs=example_outputs, 
                                                                  max_input_length=max_input_length, 
                                                                  **generator_args)
                        expl_components = preds_basic_filter(curr_completions[0]['0']['raw'], min_length=filter_min_len, remove_strings=filter_remove_strings)
                        curr_out[tem][ner] = expl_components
            outlist[t]['noun'] = copy.deepcopy(curr_out)                          
                
        if add_verb != []:
            vphrases = text_processing.verb_chunks(test_questions)
            tem = 'general'
            curr_out = {tem:{}}
            for vp in vphrases:
                question = question_templates[tem].replace('{phrase}', vp)
                if verbose:
                    print(f"Q:{t} QT:{tem} VP: {vp} Question:{question}...")
                    curr_completions = generate_continuations(templates[0], model, tokenizer, 
                                                              question, verbose=verbose, lower=lower,
                                                              example_inputs=example_inputs, example_outputs=example_outputs, 
                                                              max_input_length=max_input_length, 
                                                              **generator_args)
                    expl_components = preds_basic_filter(curr_completions[0]['0']['raw'], min_length=filter_min_len, remove_strings=filter_remove_strings)
                    curr_out[tem][vp] = expl_components
            outlist[t]['verb'] = copy.deepcopy(curr_out)                          
                
        for d in range(gen_depth):
            if verbose:
                print(f"{t} Generating explanation components at depth: {d} ...")
            if d == 0:
                test_questions = outlist[t]['question']
            else:  # d > 0
                test_questions = outlist[t]['expl_depth'][d-1] #list of 'questions' being prior generated completions

            #Returns [ { 'template_idx': 'raw': ['output 0', 'output 1', ..., 'output 9'] } ] where each row idx corresponds to input [test_questions] idx
            curr_completions = generate_continuations(templates, model, tokenizer, test_questions, verbose=verbose, lower=lower,
                                                      example_inputs=example_inputs, example_outputs=example_outputs, 
                                                      max_input_length=max_input_length, 
                                                      **generator_args)
            if verbose:
                print(f"{t} Combining preds from each template together and performing basic filtering...")
            combo = []
            for i, c in enumerate(curr_completions):
                for tem in c.keys():  #combine outputs from each input template
                    combo += c[tem]['raw']
            expl_components = preds_basic_filter(combo, min_length=filter_min_len, remove_strings=filter_remove_strings)
            # soft unique here
            if su_stage_1_stop != -1:
                expl_components = make_soft_unique(expl_components, stop_when_lessthan=su_stage_1_stop, verbose=verbose)
            outlist[t]['expl_depth'].append(expl_components)
            
        if outfile is not None:
            utils.saveas_jsonl([outlist[t]], outfile, initial_mode=initial_mode, verbose=verbose)
            initial_mode = 'a'
            
    return outlist







