"""
Adapted from https://github.com/neubig/util-scripts/paired-bootstrap.py
by Tim Hartill
"""

######################################################################
# Compare two systems using bootstrap resampling                     #
#  * by Graham Neubig                                                #
#  * minor modifications by Mathias Müller                           #
#                                                                    #
# See, e.g. the following paper for references                       #
#                                                                    #
# Statistical Significance Tests for Machine Translation Evaluation  #
# Philipp Koehn                                                      #
# http://www.aclweb.org/anthology/W04-3250                           #
#                                                                    #
######################################################################

import os
import json
import logging

import numpy as np

import utils
import eval_metrics


EVAL_TYPE_ACC = "acc"
EVAL_TYPE_BLEU = "bleu"
EVAL_TYPE_BLEU_DETOK = "bleu_detok"
EVAL_TYPE_PEARSON = "pearson"
# TJH Added:
EVAL_TYPE_YN = 'yn'
EVAL_TYPE_F1 = 'f1'  # DROP-style f1 where if there is a number in the gold answer and it doesnt match then F1=0 regardless of other matches, otherwise "normal" F1   
EVAL_TYPE_MC = 'mc'


EVAL_TYPES = [EVAL_TYPE_ACC, EVAL_TYPE_BLEU, EVAL_TYPE_BLEU_DETOK, EVAL_TYPE_PEARSON,
              EVAL_TYPE_YN, EVAL_TYPE_F1, EVAL_TYPE_MC]


def eval_preproc(data, eval_type='acc'):
    ''' Preprocess into the appropriate format for a particular evaluation type '''
    if type(data) == str:
        data = data.strip()
        if eval_type == EVAL_TYPE_BLEU:
            data = data.split()
        elif eval_type == EVAL_TYPE_PEARSON:
            data = float(data)
    return data


def eval_measure(gold, sys, questions=[], eval_type='acc'):
    ''' Evaluation measure
    
    This takes in gold labels and system outputs and evaluates their
    accuracy. It currently supports:
    * Accuracy (acc), percentage of labels that match
    * Pearson's correlation coefficient (pearson)
    * BLEU score (bleu)
    * BLEU_detok, on detokenized references and translations, with internal tokenization
    
    * YN, Drop-style F1, Multichoice
    
    :param gold: [the correct labels]
    :param sys: [the system outputs]
    :param questions = [question list in tsv format for extracting MC options from]
    :param eval_type: The type of evaluation to do (acc, pearson, bleu, bleu_detok etc)
    '''
    if eval_type == EVAL_TYPE_ACC:
        return sum([1 if g == s else 0 for g, s in zip(gold, sys)]) / float(len(gold))
    elif eval_type == EVAL_TYPE_BLEU:
        import nltk
        gold_wrap = [[x] for x in gold]
        return nltk.translate.bleu_score.corpus_bleu(gold_wrap, sys)
    elif eval_type == EVAL_TYPE_PEARSON:
        return np.corrcoef([gold, sys])[0,1]
    elif eval_type == EVAL_TYPE_BLEU_DETOK:
        import sacrebleu
        # make sure score is 0-based instead of 100-based
        return sacrebleu.corpus_bleu(sys, [gold]).score / 100.
    elif eval_type == EVAL_TYPE_YN:
        calcobj = eval_metrics.YN()
        return calcobj.compute_metric(sys, gold) / 100. , calcobj 
    elif eval_type == EVAL_TYPE_F1:
        calcobj = eval_metrics.F1()  
        return calcobj.compute_metric(sys, gold) / 100. , calcobj
    elif eval_type == EVAL_TYPE_MC:
        calcobj = eval_metrics.StringSimilarity()  
        return calcobj.compute_metric(sys, gold, questions) / 100. , calcobj
        
    else:
        raise NotImplementedError('Unknown eval type in eval_measure: %s' % eval_type)
    return


def eval_with_paired_bootstrap(logger, gold, sys1, sys2, questions=[],
                               num_samples=10000, sample_ratio=0.5,
                               eval_type='acc'):
    ''' Evaluate with paired boostrap
    
    This compares two systems, performing a significance tests with
    paired bootstrap resampling to compare the accuracy of the two systems.
    
    :param gold: [The correct labels]
    :param sys1: [The output of system 1]
    :param sys2: [The output of system 2]
    :param questions = [questions in tsv format for extracting MC options from]
    :param num_samples: The number of bootstrap samples to take
    :param sample_ratio: The ratio of samples to take every time
    :param eval_type: The type of evaluation to do (acc, pearson, bleu, bleu_detok)
    '''
    assert(len(gold) == len(sys1))
    assert(len(gold) == len(sys2))
    
    # Preprocess the data appropriately for they type of eval
    gold = [eval_preproc(x, eval_type) for x in gold]
    sys1 = [eval_preproc(x, eval_type) for x in sys1]
    sys2 = [eval_preproc(x, eval_type) for x in sys2]
    
    # precalculate all scores for faster processing where possible
    if eval_type in [EVAL_TYPE_F1, EVAL_TYPE_YN, EVAL_TYPE_MC]:
        sys1_overall_score, sys1_calcobj = eval_measure(gold, sys1, questions, eval_type=eval_type)
        sys2_overall_score, sys2_calcobj = eval_measure(gold, sys2, questions, eval_type=eval_type)
    else:
        sys1_overall_score, sys2_overall_score = -1.0, -1.0  
    
    sys1_scores = []
    sys2_scores = []
    wins = [0, 0, 0]
    diff_scores = []
    n = len(gold)
    ids = list(range(n))
    
    for i in range(num_samples):
        # Subsample the gold and system outputs
        reduced_ids = np.random.choice(ids,int(len(ids)*sample_ratio), replace=True)
        if sys1_overall_score != -1:  # use precalculated scores if possible
            sys1_score = sys1_calcobj.all_scores[reduced_ids].mean()
            sys2_score = sys2_calcobj.all_scores[reduced_ids].mean()
        else:
            reduced_gold = [gold[i] for i in reduced_ids]
            reduced_sys1 = [sys1[i] for i in reduced_ids]
            reduced_sys2 = [sys2[i] for i in reduced_ids]
            if eval_type == EVAL_TYPE_MC:
                reduced_questions = [questions[i] for i in reduced_ids]
            else:
                reduced_questions = []
            # Calculate accuracy on the reduced sample and save stats
            sys1_score = eval_measure(reduced_gold, reduced_sys1, reduced_questions, eval_type=eval_type)
            sys2_score = eval_measure(reduced_gold, reduced_sys2, reduced_questions, eval_type=eval_type)
        if sys1_score > sys2_score:
            wins[0] += 1
        elif sys1_score < sys2_score:
            wins[1] += 1
        else:
            wins[2] += 1
        sys1_scores.append(sys1_score)
        sys2_scores.append(sys2_score)
        diff = sys2_score - sys1_score
        diff_scores.append(diff)
        
    # Print win stats
    wins = [x/float(num_samples) for x in wins]
    logger.info('Win ratio: sys1=%.3f, sys2=%.3f, tie=%.3f' % (wins[0], wins[1], wins[2]))
    if wins[0] > wins[1]:
        logger.info('(sys1 is superior with p value p=%.3f)\n' % (1-wins[0]))
    elif wins[1] > wins[0]:
        logger.info('(sys2 is superior with p value p=%.3f)\n' % (1-wins[1]))
    
    # Print system stats
    sys1_scores.sort()
    sys2_scores.sort()
    diff_scores.sort()
    logger.info('sys1 mean=%.3f, median=%.3f, 95%% confidence interval=[%.3f, %.3f]' %
            (np.mean(sys1_scores), np.median(sys1_scores), sys1_scores[int(num_samples * 0.025)], sys1_scores[int(num_samples * 0.975)]))
    logger.info('sys2 mean=%.3f, median=%.3f, 95%% confidence interval=[%.3f, %.3f]' %
            (np.mean(sys2_scores), np.median(sys2_scores), sys2_scores[int(num_samples * 0.025)], sys2_scores[int(num_samples * 0.975)]))
    logger.info('DIFF sys2-sys1 mean=%.3f, median=%.3f, 95%% confidence interval=[%.3f, %.3f]' %
            (np.mean(diff_scores), np.median(diff_scores), diff_scores[int(num_samples * 0.025)], diff_scores[int(num_samples * 0.975)]))
    
    logger.info('sys1 mean=%.3f, median=%.3f, 90%% confidence interval=[%.3f, %.3f]' %
            (np.mean(sys1_scores), np.median(sys1_scores), sys1_scores[int(num_samples * 0.05)], sys1_scores[int(num_samples * 0.95)]))
    logger.info('sys2 mean=%.3f, median=%.3f, 90%% confidence interval=[%.3f, %.3f]' %
            (np.mean(sys2_scores), np.median(sys2_scores), sys2_scores[int(num_samples * 0.05)], sys2_scores[int(num_samples * 0.95)]))
    logger.info('DIFF sys2-sys1 mean=%.3f, median=%.3f, 90%% confidence interval=[%.3f, %.3f]' %
            (np.mean(diff_scores), np.median(diff_scores), diff_scores[int(num_samples * 0.05)], diff_scores[int(num_samples * 0.95)]))   
    return


if __name__ == "__main__":
    # execute only if run as a script
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', type=str, help='File of the correct answers in tsv format')
    parser.add_argument('--sys1', type=str, help='File of the answers for system 1')
    parser.add_argument('--sys2', type=str, help='File of the answers for system 2')
    parser.add_argument('--eval_type', help='The evaluation type (acc/pearson/bleu/bleu_detok/yn/f1/mc)', type=str, default='acc', choices=EVAL_TYPES)
    parser.add_argument('--num_samples', help='Number of sample means to use', type=int, default=10000)
    parser.add_argument('--sample_ratio', help='Percentage of overall samples to take per sample mean', type=float, default=1.0)
    parser.add_argument('--output_file', type=str, help='output file')
    args = parser.parse_args()
    
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
          datefmt='%m/%d/%Y %H:%M:%S',
          level=logging.INFO,
          handlers=[logging.FileHandler(args.output_file),
                    logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)

    if not os.path.exists(args.gold):
        logger.info(f'ERROR: gold path doesnt exist: {args.gold}')
    if not os.path.exists(args.sys1):
        logger.info(f'ERROR: sys1 path doesnt exist: {args.sys1}')
    if not os.path.exists(args.sys2):
        logger.info(f'ERROR: sys2 path doesnt exist: {args.sys2}')
    
    questions, gold = utils.load_uqa_supervised(args.gold, ans_lower=True)
    
    sys1 = json.load(open(args.sys1))
    sys2 = json.load(open(args.sys2))
    #  with open(args.gold, 'r') as f:
    #    gold = f.readlines() 
    #  with open(args.sys1, 'r') as f:
    #    sys1 = f.readlines() 
    #  with open(args.sys2, 'r') as f:
    #    sys2 = f.readlines() 
    eval_with_paired_bootstrap(logger, gold, sys1, sys2, questions, eval_type=args.eval_type, 
                               num_samples=args.num_samples, sample_ratio=args.sample_ratio)
