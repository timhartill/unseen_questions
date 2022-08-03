#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 2022

@author: tim hartill

Adapted from Geva et al 2020 "Injecting numerical reasoning skills into language models" gen_numeric_data.py 
to generate large amounts of synthetic data with POET-style variablised context

Edit indir and outdir before running...

Note: first run:
import nltk 
nltk.download('words')


"""


import uuid, random, jsonlines, logging, argparse, re
from datetime import datetime, date, timedelta
from dateutil import relativedelta
from random import shuffle

import ujson as json
from tqdm import tqdm
import numpy as np
from nltk.corpus import words, wordnet
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large', do_lower_case=True)

var_mapping = [chr(i) for i in list(range(97, 123))] + [chr(i)+chr(i) for i in list(range(97, 123))]  # BART tokenizer parse all these to single toks

# words with <= 2 tokens
nltk_words = [w.lower() for w in words.words() if len(tokenizer.tokenize(w)) <= 2 and w not in set(var_mapping)]  #64944 words

context_vars = 30
max_num = 20000

superlatives = {'max':["longest", "last", "highest", "largest", "most", "greatest"], 
                'min':["shortest", "first", "smallest", "lowest", "least"]}

argsuperlatives = {'argmax':["highest", "largest", "biggest", "maximum", "greatest"], 
                   'argmin':["minimum", "smallest", "lowest", "least"]}

comparatives = {'less': ["fewer", "less", "before", "earlier", "smaller", "lower", "shorter"],
                'more': ["more", "later", "bigger", "higher", "longer", "larger", "greater", "taller"]}

date_superlatives = {'max':["last", "latest", "most recent", "youngest"], 
                    'min':["first", "earliest", "oldest", 'least recent']}

date_comparatives = {'less': ["before", "earlier", "older"],
                     'more': ['after', "later", "younger"]}


# def rand_expression(args):
#     # returns arithmetic expression, val
#     if len(args) == 1:
#         return str(args[0]), args[0]
#     split = random.randint(1, len(args)-1)
#     exp1, val1 = rand_expression(args[:split])
#     exp2, val2 = rand_expression(args[split:])
#     op = random.choice(['+', '-', '*']) #if len(str(val1*val2)) < 10 else random.choice(['+', '-'])
#     val = {'+': val1 + val2, '-': val1 - val2, '*': val1 * val2}[op]
#     return '(%s %s %s)' % (exp1, op, exp2), val # infix


def fill_context(i, subst, valtype='num'):
    """ Fill the context with distractors
    """
    while i < context_vars-1:
        v = var_mapping[i]
        if valtype == 'num':
            n = random.randint(0, max_num)
            if random.randint(0,1):  # 50% prob
                n = rand_float(n)
        elif valtype == 'pc':
            n = random.randint(0, 100)
            if random.randint(0,1):  # 50% prob
                n = rand_float(n)
            n = str(n) + '%'            
        else:  # 'date'
            n = datetime.now() - timedelta(days=2018*365) * random.random()
            day_diff_range = 30 if random.randint(0,1) else 150
            diff = random.sample(range(1, day_diff_range+1), 1)[0]
            n = n + random.choice([-1,1]) * timedelta(days=diff)
            n = [n.strftime("%d %B %Y"), n.strftime("%B %d, %Y")][random.randint(0, 1)]            
        subst.append(f"{v} = {str(n)} ;")
        i += 1
    return

def rand_float(x):
    # randomly add up to 2 decimal places
    precision = np.random.choice([0, 1, 2], p=[0.2, 0.4, 0.4])
    fractional_part = {0: 0, 1: random.randint(0, 9)*0.1, 2: random.randint(0, 99)*0.01}[precision]
    return x + fractional_part


def signed_expression(args):
    """ returns signed combination, val, context
    eg ('x - y - z', -4.4, "a = 2 ; x = 1.1 ; z = 2.2 ; y = 3.3 ; ff = 12.4 ;")
    """
    shuffle(var_mapping)
    expr, val = '', 0
    subst = []
    i = 0
    for a in args:
        v = var_mapping[i]
        subst.append(f"{v} = {str(a)} ;")
        sign = random.choice(['+', '-']) 
        val += {'+': a, '-': -a}[sign]
        expr += '%s %s ' % (sign, str(v))
        i += 1
    fill_context(i, subst)
    expr = expr[1:] if expr[0] == '+' else expr
    shuffle(subst)
    return "What is " + expr.strip() + '?', round(val, 2), ' '.join(subst)


def min_max_avg_expression(args):
    """ returns min/max expression, val, context
    eg ('average(1.1, 2.2, 3.3)', 2.2, ..)
    """
    shuffle(var_mapping)
    expr, val = '', 0
    choice = random.randint(0,2)
    val = [max(args), min(args), round(sum(args)/len(args), 2)][choice]
    subst = []
    i = 0
    for a in args:
        v = var_mapping[i]
        subst.append(f"{v} = {str(a)} ;")
        expr += v + ', '
        i += 1
    fill_context(i, subst)
    expr = expr[:-2]    
    #expr = ', '.join(map(str, args)).strip()
    expr = 'What is the %s of %s?' % ([random.choice(superlatives['max']), random.choice(superlatives['min']), 
                        'average'][choice], expr)
    shuffle(subst)
    return expr.strip(), val, ' '.join(subst)


def arg_min_max_expression(wrds, args):
    """ returns argmin/argmax expression, val
    eg ('argmin(word1 1.1, word2 2.2, word3 3.3)', 'word1')
    """
    shuffle(var_mapping)
    expr = ''
    subst = []
    i = 0
    for w, a in zip(wrds, args):
        v = var_mapping[i]
        subst.append(f"{v} = {str(a)} ;")
        expr += '%s %s, ' % (w, str(v))
        i += 1
    fill_context(i, subst)
    mn, mx, expr = min(args), max(args), expr[:-2].strip()
    max_or_min = random.randint(0,1)
    val = wrds[args.index(mx)] if max_or_min else wrds[args.index(mn)]
    expr = 'What is the element with %s value of: %s?' % (random.choice(argsuperlatives['argmax']) if max_or_min else random.choice(argsuperlatives['argmin']), expr)
    shuffle(subst)
    return expr.strip(), val, ' '.join(subst)


def rand_percent():
    """
    # returns expression, val, args
    # sample 3-5 args
    ('percent not marla :: marla 14%, unrid 18%, antu 68%',
     'marla 14%, unrid 18%, antu 68%',
     'percent not marla',
     86,
     [14, 18, 68])
    """
    shuffle(var_mapping)
    wrds = [random.choice(nltk_words)
            for _ in range(np.random.choice([3, 4, 5], p=[0.2, 0.4, 0.4]))]
    args = []
    for p in np.random.dirichlet(np.ones(len(wrds)))*100:
        p = {0:float, 1: int}[random.randint(0,1)]((round(p, random.randint(1,2))))
        args.append(p)
    args[0] = round(100 - sum(args[1:]), 2)
    context = ''
    subst = []
    i = 0    
    for w, a in zip(wrds, args):
        v = var_mapping[i]
        subst.append(f"{v} = {str(a)}% ;")
        context += '%s %s, ' % (w, str(v))
        i += 1
    fill_context(i, subst, valtype='pc')
    context = context[:-2].strip()
    n_q_args = min(np.random.choice([1, 2, 3], p=[0.4, 0.3, 0.3]), len(args) - 1)
    q_ids_wrds = random.sample(list(enumerate(wrds)), n_q_args)
    q_args, q_wrds = [], []
    for tup in q_ids_wrds:
        q_args.append(args[tup[0]]); q_wrds.append(tup[1])
    negate = random.choice(['', 'not '])
    q = 'What percentage is %s' % negate + ', '.join(q_wrds)
    expr = q + ' of: ' + context + '?'
    val = {'': sum(q_args), 'not ': 100 - sum(q_args)}[negate]
    shuffle(subst)
    return expr.strip(), context.strip(), q.strip(), round(val, 2), args, ' '.join(subst)


def date_min_max(n_args=3):
    """ returns min/max expression, val, args
    eg ('latest(March 09, 1887; 19 March 1887; July 16, 1887)', 'July 16, 1887',
        ['March 09, 1887', '19 March 1887', 'July 16, 1887'])
    """
    shuffle(var_mapping)
    rds = [datetime.now() - timedelta(days=2018*365) * random.random() for _ in range(n_args)]
    day_diff_range = 30 if random.randint(0,1) else 150
    diffs = random.sample(range(1, day_diff_range+1), n_args-1)
    for i in range(1, len(rds)):
        rds[i] = rds[0] + random.choice([-1,1]) * timedelta(days=diffs[i-1])
    random.shuffle(rds)
    choices = [[rd.strftime("%d %B %Y"), rd.strftime("%B %d, %Y")][random.randint(0, 1)] for rd in rds]
    max_or_min = random.randint(0,1)
    expr = ''
    subst = []
    i = 0
    for a in choices:
        v = var_mapping[i]
        subst.append(f"{v} = {str(a)} ;")
        expr += v + ', '
        i += 1
    fill_context(i, subst, valtype='date')
    expr = expr[:-2]    
    
    #expr = '; '.join(choices).strip()
    rd = [max(rds), min(rds)][max_or_min]
    val = choices[rds.index(rd)]
    expr = 'What is the %s of: %s?' % ([random.choice(date_superlatives['max']), 
                        random.choice(date_superlatives['min'])][max_or_min], expr)
    shuffle(subst)
    return expr.strip(), val, choices, ' '.join(subst)


def date_diff(typ=''):
    """ returns expression, val, args
    eg ('difference in days(02 October 736; 26 April 736)', 159,
        ['02 October 736', '26 April 736'])
    """
    shuffle(var_mapping)
    typ = typ if typ else random.choice(['years', 'months', 'days'])
    rds = [datetime.now() - timedelta(days=2018*365) * random.random() for _ in range(2)]
    if typ in ['months', 'days']:
        diff = timedelta(days=60) if random.randint(0,1) else timedelta(days=200)
        rds[1] = rds[0] + random.choice([-1,1]) * diff * random.random()
    random.shuffle(rds)
    choices = [[rd.strftime("%d %B %Y"), rd.strftime("%B %d, %Y")][random.randint(0, 1)] for rd in rds]
    # DROP: yr diff depends only on yr vals, similarly for months within an yr
    diff_years = max(rds).year - min(rds).year
    diff_months = diff_years*12 + (max(rds).month - min(rds).month)
    diff_days = (max(rds) - min(rds)).days
    val = {'years':diff_years, 'months':diff_months, 'days':diff_days}[typ]
    expr = ''
    subst = []
    i = 0
    for a in choices:
        v = var_mapping[i]
        subst.append(f"{v} = {str(a)} ;")
        expr += v + ', '
        i += 1
    fill_context(i, subst, valtype='date')
    expr = expr[:-2]    

    #expr = '; '.join(choices).strip()
    expr = 'What is the difference in %s of %s?' % (typ, expr)
    shuffle(subst)
    return expr.strip(), val, choices, ' '.join(subst)


def main():
    parser = argparse.ArgumentParser(description='For generating variablized synthetic numeric data.')
    parser.add_argument("--num_samples", default=1e6, type=float, help="Total number of samples to generate.")
    parser.add_argument("--num_dev_samples", default=1e4, type=float, help="Num of samples to keep aside for dev set.")
    parser.add_argument("--output_jsonl", default='./data/synthetic_numeric.jsonl', type=str, 
                        help="Output synthetic numeric data .jsonl file.")
    pargs = parser.parse_args()
    
    # split the domain
    domain, train_number_range, dev_number_range = int(max_num), [], []
    for i in range(domain):
        x = train_number_range if random.random() < 0.8 else dev_number_range
        x.append(i)

    n_examples, n_dev, q_types = int(pargs.num_samples), int(pargs.num_dev_samples), 6
    discrete_ops_data, n_iters = [], n_examples // q_types
    train_args, dev_args = set(), set()

    print(f"Creating {n_examples} samples...")
    for i_s in tqdm(range(n_iters)):
        # decide train/dev split
        split = 'train' if i_s < n_iters - (n_dev // q_types) else 'dev'
        rng = {'train': train_number_range, 'dev': dev_number_range}[split]
        args = [random.choice(rng) for _ in range(np.random.choice([2, 3, 4], p=[1/3]*3))]
        # with 50% prob add rand fraction
        args = list(map(rand_float, args)) if random.randint(0,1) else args
        train_args.update(args) if split == 'train' else dev_args.update(args)

        wrds = [random.choice(nltk_words) for _ in range(len(args))]

        expr, val, context = signed_expression(args)
        d1 = {'id': str(uuid.uuid4().hex), 'expr': expr, 'val': val, 'args': args, 
              'type': 'signed_expression', 'check_domain':True, 'split': split, context: context}

        expr, val, context = min_max_avg_expression(args)
        d2 = {'id': str(uuid.uuid4().hex), 'expr': expr, 'val': val, 'args': args, 
              'type': 'min_max_avg_expression', 'check_domain':True, 'split': split, context: context}

        expr, val, context = arg_min_max_expression(wrds, args)
        d3 = {'id': str(uuid.uuid4().hex), 'expr': expr, 'val': val, 'args': args, 
              'type': 'arg_min_max_expression', 'check_domain':True, 'split': split, context: context}

        expr, val, date_args, context = date_min_max(n_args=len(args))
        d4 = {'id': str(uuid.uuid4().hex), 'expr': expr, 'val': val, 'args': date_args, 
              'type': 'date_min_max', 'check_domain':False, 'split': split, context: context}

        expr, val, date_args, context = date_diff()
        d5 = {'id': str(uuid.uuid4().hex), 'expr': expr, 'val': val, 'args': date_args, 
              'type': 'date_diff', 'check_domain':False, 'split': split, context: context}

        expr, oldcontext, qn, val, args, context = rand_percent()
        d6 = {'id': str(uuid.uuid4().hex), 'expr': expr, 'val': val, 'args': args, 'ques': qn, 
              'context': context, 'type': 'percent', 'check_domain':False, 'split': split}

        discrete_ops_data += [d1, d2, d3, d4, d5, d6]

    assert train_args.isdisjoint(dev_args) # trn, dev args are disjoint

    with jsonlines.open(pargs.output_jsonl, mode='w') as writer:
        writer.write_all(discrete_ops_data)
    

if __name__ == "__main__":
    main()
    
    
'''
python gen_numeric_data.py --num_samples 1e6 --num_dev_samples 1e4 --output_jsonl ../data/synthetic_numeric.jsonl
'''