"""
F1, n-gram and sentence embedding test-train overlap detector
Also contains the functions to run reports combining test-train similarity and prediction performance
To run the reports use the run_all_reports(...) function below. It is designed to be run interactively 
rather than from the command line.

Author: Tim Hartill

The n-gram overlap detector is adapted from code from: https://github.com/elangovana/nlp-train-test-overlap-detector

"""

from typing import List
import os
import json
import math
import copy
import collections
import datetime

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import eval_metrics


class OverlapDetectorEmbedding:
    """ Compare embeddings for source vs target rows for each column in [columns]
        Return most similar match (as measured by sentence embedding similarity) for each source row in target for each column 
    """
    def __init__(self, similarity_comparer, train_emb, test_emb, ssvise): 
        self._similarity_comparer = similarity_comparer
        self.train_emb = train_emb
        self.test_emb = test_emb
        self.ssvise = ssvise

    def compare(self, source_df, target_df, columns=None, add_combo=True, answer_thresh=-100.1, answer_col='label', question_thresh=0.0):
        columns = columns or source_df.columns

        result = {}
        similarity_scores = {}
        for c in columns:
            if c != answer_col:
                question_col = c
                break

        for c in columns:
            result[c] = {}
            if not self.ssvise or c != answer_col:
                result[c]["score"], result[c]["details"], similarity_scores[c] = self._compare_rows_text(source_df[c].tolist(),
                                                                                                         target_df[c].tolist(), c )
            else:
                result[c]["score"] = []
                result[c]["details"] = []
                
        if add_combo: # add a combined score based on similar answers with similar questions in same training example
            if not self.ssvise:
                answer_sims = similarity_scores[answer_col] # (n_samples_test, n_samples_train)
                answer_mask = (answer_sims >= answer_thresh)
                divisor = 1.0 * len(similarity_scores)
                combo_similarity = np.zeros(answer_sims.shape)
                for c in similarity_scores:
                    if c != answer_col:
                        question_mask = (similarity_scores[c] >= question_thresh)
                        similarity_scores[c] *= question_mask
                    similarity_scores[c] *= answer_mask  # if answer similarity < threshold, overall similarity = 0
                    combo_similarity += similarity_scores[c]
                combo_similarity /= divisor
                combo_result_score = np.max(combo_similarity, axis=1).tolist()      # compute most similar target to each source [n_samples_test]
                combo_similarity_scores_index = np.argmax(combo_similarity, axis=1) # [n_samples_test]
                combo_result_detailed_match = []
                for i, mi in enumerate(combo_similarity_scores_index):
                    test_sample = ''
                    train_sample = ''
                    for c in similarity_scores:
                        test_sample += c + ': ' + source_df.iloc[i][c] + ' '
                        train_sample += c + ': ' + target_df.iloc[mi][c] + ' '
                    combo_result_detailed_match.append( (test_sample.strip(), train_sample.strip()) )
                result['combo'] = {}
                result['combo']['score'] = combo_result_score
                result['combo']['details'] = combo_result_detailed_match
            else:                       # ssvise so combo just = question similarity
                combo_result_detailed_match = []
                combo_similarity_scores_index = np.argmax(similarity_scores[question_col], axis=1) # [n_samples_test]
                for i, mi in enumerate(combo_similarity_scores_index):
                    test_sample = result[question_col]["details"][i][0] + ' ' + answer_col + ': ' + source_df.iloc[i][answer_col]
                    train_sample = result[question_col]["details"][i][1] + ' ' + answer_col + ': ' + target_df.iloc[mi][answer_col]
                    combo_result_detailed_match.append( (test_sample.strip(), train_sample.strip()) )                
                result['combo'] = {}
                result['combo']['score'] = result[question_col]["score"]
                result['combo']['details'] = combo_result_detailed_match
                                        
        return result

    def _compare_rows_text(self, src_rows, target_rows, c):
        
        similarity_score = self._similarity_comparer(self.test_emb[c], self.train_emb[c])  # returns shape (n_samples_test, n_samples_train)

        result_score = np.max(similarity_score, axis=1).tolist()        # compute most similar target to each source [n_samples_test]
        similarity_scores_index = np.argmax(similarity_score, axis=1)   # compute index of most similar target to each source [n_samples_test]
        result_detailed_match = [(src_rows[i], target_rows[mi]) for i, mi in enumerate(similarity_scores_index)]  #n_samples_test sized list of (test example text, train example text)

        return result_score, result_detailed_match, similarity_score


class CosineSimilarityEmbeddingComparer:
    """ Calculate cosine similarity of each source embedding to each target embedding
        source: np array of shape [num_source, 1024]
        target: np array of shape [num_target, 1024]
        Note: computing cos sim for [22k,1024] against [2.6m,1024] for 'text' then 'label' exceeds 768GB ram...
    """
    def __init(self):
        pass
    
    def __call__(self, source, target):
        # returns shape (n_samples_X, n_samples_Y)
        similarity_score = cosine_similarity(source, Y=target, dense_output=True)
        return similarity_score * 100
    

class OverlapDetector:
    """ Compare source vs target rows for each column in [columns]
        Return most similar match (as measured by BOW word overlap) for each source row in target for each column 
    """
    def __init__(self, similarity_comparer, ssvise): 
        self._similarity_comparer = similarity_comparer
        self.ssvise = ssvise

    def compare(self, source_df, target_df, columns=None, add_combo=True, answer_thresh=40.0, answer_col='label', question_thresh=0.0):
        columns = columns or source_df.columns

        result = {}
        similarity_scores = {}
        for c in columns:
            if c != answer_col:
                question_col = c
                break
            
        for c in columns:
            result[c] = {}
            if not self.ssvise or c != answer_col:
                result[c]["score"], result[c]["details"], similarity_scores[c] = self._compare_rows_text(source_df[c].tolist(),
                                                                                                         target_df[c].tolist())
            else:
                result[c]["score"] = []
                result[c]["details"] = []
        if add_combo: # add a combined score based on similar answers with similar questions in same training example
            if not self.ssvise:
                answer_sims = similarity_scores[answer_col]
                answer_mask = (answer_sims >= answer_thresh)
                divisor = 1.0 * len(similarity_scores)
                combo_similarity = np.zeros(answer_sims.shape)
                for c in similarity_scores:
                    if c != answer_col:
                        question_mask = (similarity_scores[c] >= question_thresh)
                        similarity_scores[c] *= question_mask
                    similarity_scores[c] *= answer_mask  # if answer similarity < threshold, overall similarity = 0
                    combo_similarity += similarity_scores[c]
                combo_similarity /= divisor
                combo_result_score = np.max(combo_similarity, axis=1).tolist()  # compute most similar target to each source
                combo_similarity_scores_index = np.argmax(combo_similarity, axis=1)
                combo_result_detailed_match = []
                for i, mi in enumerate(combo_similarity_scores_index):
                    test_sample = ''
                    train_sample = ''
                    for c in similarity_scores:
                        test_sample += c + ': ' + source_df.iloc[i][c] + ' '
                        train_sample += c + ': ' + target_df.iloc[mi][c] + ' '
                    combo_result_detailed_match.append( (test_sample.strip(), train_sample.strip()) )
                result['combo'] = {}
                result['combo']['score'] = combo_result_score
                result['combo']['details'] = combo_result_detailed_match
            else:                       # ssvise so combo just = question similarity
                result['combo'] = {}
                result['combo']['score'] = result[question_col]["score"]
                result['combo']['details'] = result[question_col]["details"]
                  
        return result

    def _compare_rows_text(self, src_rows, target_rows):
        similarity_score = self._similarity_comparer(src_rows, target_rows)

        result_score = np.max(similarity_score, axis=1).tolist()  # compute most similar target to each source
        similarity_scores_index = np.argmax(similarity_score, axis=1)
        result_detailed_match = [(src_rows[i], target_rows[mi]) for i, mi in enumerate(similarity_scores_index)]  #(test example, train example)

        return result_score, result_detailed_match, similarity_score


class CosineSimilarityComparer:
    """ Calculate n-gram counts of source and target rows as vectors then return 
        cosine similarity of each source to each target. 
        Note: stop_words updated from default 'english' since 'no' and other important words were included.
    """
    def __init__(self, ngram, stop_words='special', vectoriser='count'):
        # here remove words like 'no' that are a key part of certain training examples.. set  stop_words='english' for default behaviour
        if stop_words == 'special':
            stop_words = ['off',  'by', 'sincere', 'per', 'hers', 'only', 'take', 'so', 'the', 'their', 'another', 'how', 'else', 'several', 'with', 'them', 'whether',
                          'hereupon', 'something', 'nowhere', 'to', 'own', 'anything', 'nothing', 'seemed', 'much', 'noone', 'wherever', 'whose',  'thence', 'they',
                          'herein', 'whither', 'amount', 'bill', 'co', 'your', 'herself', 'hereafter', 'well', 'made', 'therein', 'is', 'about', 'ours',
                          'should', 'was', 'etc', 'there', 'for', 'thereby', 'whereby', 'becoming', 'also', 'during', 'ourselves', 'such', 'rather',
                          'give', 'be', 'nobody', 'though', 'myself', 'hereby', 'why', 'being', 'elsewhere', 'meanwhile', 'detail', 'mine', 'could',
                          'couldnt', 'found', 'indeed', 'a', 'which', 'moreover', 'than', 'further', 'can', 'somewhere', 'both', 'thru', 'sometime',
                          'everyone', 'wherein', 'hence', 'mostly', 'amoungst', 'anyhow', 'yours', 'serious', 'beyond', 'side', 'across',
                          'still', 'whenever', 'we', 'without', 'alone', 'am', 'show', 'although', 'were', 'inc', 'name', 'seem', 'namely',
                         'ie', 'when', 'most', 'anywhere', 'him', 'whereas', 'will', 'already', 'last', 're', 'until', 'sometimes', 'i', 'upon', 'see', 'besides', 'us', 'our', 'otherwise',
                         'thereupon', 'whatever', 'each', 'very', 'beforehand', 'whom', 'un', 'becomes', 'due', 'often', 'its', 'interest', 'yet', 'almost', 'cry', 'must', 'latterly',
                         'go', 'her', 'been', 'call', 'she', 'too', 'into', 'become' 'whence', 'what', 'are', 'my', 'as', 'eg', 'always', 'find', 'con', 'perhaps',
                         'yourselves', 'while', 'through', 'formerly', 'then', 'where', 'nevertheless', 'others', 'therefore', 'himself', 'describe', 'do', 'again', 'along',
                         'beside', 'fill', 'he', 'amongst', 'however', 'ltd', 'of', 'became', 'mill', 'anyway', 'whereafter', 'here', 'since', 'at', 'an', 'themselves', 'these', 'those',
                         'it', 'on', 'done', 'keep', 'whereupon', 'that', 'seems', 'around', 'his', 'but', 'has', 'yourself', 'before', 'somehow', 'within', 'thus', 'other',
                         'thereafter', 'via', 'seeming', 'had', 'ever', 'whoever', 'move', 'because', 'among', 'itself', 'throughout', 'who', 'may', 'you', 'would',
                         'enough', 'someone', 'de', 'this', 'me', 'onto', 'have', 'put']
        if vectoriser == 'count':
            self._vectoriser = CountVectorizer(stop_words=stop_words, ngram_range=(ngram, ngram))
        else:
            self._vectoriser = TfidfVectorizer(stop_words=stop_words, ngram_range=(ngram, ngram))

    def __call__(self, source: List[str], target: List[str]):
        try:
            vectoriser = self._vectoriser.fit(source + target)
        # Only stop words
        except ValueError as e:
            print(e)
            return np.zeros((len(source), len(target)))  #.tolist()

        # Calculate n-gram counts of source and target rows, returns shape [num_rows, num_n_grams] with values being counts of each ngram
        src_vector = vectoriser.transform(source)
        target_vector = vectoriser.transform(target)
        
        # returns shape (n_samples_X, n_samples_Y)
        similarity_score = cosine_similarity(src_vector, Y=target_vector, dense_output=True)
        return similarity_score * 100


class F1SimilarityComparer:
    """ Calculate F1 similarity between each source (test) and each target (train)
        Assumes input text is already normalised
    """
    def __init__(self):
        pass
    
    def _f1_faster(self, pred_toks, gold_toks, pred_count, gold_count):
        common = gold_count & pred_count
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def _tokenise(self, testlist: List[str]):
        tok_testlist = [t.split() for t in testlist]
        return tok_testlist

    def _counterize(self, test_toks):
        counter_test_toks = [collections.Counter(t) for t in test_toks]
        return counter_test_toks
    
    def __call__(self, source: List[str], target: List[str]):
        similarity_score = np.zeros((len(source), len(target)))
        print(f"{str(datetime.datetime.now())} Tokenising test data..")
        source_tok = self._tokenise(source)
        print(f"{str(datetime.datetime.now())} Tokenising train data..")
        target_tok = self._tokenise(target)
        print(f"{str(datetime.datetime.now())} Counterising test data..")
        source_count = self._counterize(source_tok)
        print(f"{str(datetime.datetime.now())} Counterising train data..")
        target_count = self._counterize(target_tok)

        print(f"Number of test samples: {len(source_tok)}")
        for i in range(len(source_tok)):
            print(f"{str(datetime.datetime.now())} Calculating f1 for test sample {i} against {len(target_tok)} train samples...")
            for j in range(len(target_tok)):
                similarity_score[i,j] = self._f1_faster(source_tok[i], target_tok[j], 
                                                        source_count[i], target_count[j])
        return similarity_score * 100
    

class SimilarityEvaluator:
    """ Evaluate similarity between equivalent column(s) in a test (source) and a train (target) df
    """
    def __init__(self, ngrams_dict=None):
        """
        :param ngrams_dict: Name and ngram. Example
                            {"Unigram": 1, "Bigram": 2, "Trigram": 3}  
        """
        self._ngrams = ngrams_dict or {"Unigram": 1}

    def run(self, test, train, column, add_combo=True, answer_thresh=40.0, 
            answer_col='label', question_thresh=0.0, use_cosine='bow', ssvise=False,
            train_emb=None, test_emb=None):
        """ Returns:
        result_score: dict keys dict_keys(['Unigram', 'Bigram', 'Trigram']) 
                    each having dict_keys(['text', 'label', 'combo'])
                    each having list of len num_test_samples each being the max similarity to the current train dataset
        result_detail: dict keys dict_keys(['Unigram', 'Bigram', 'Trigram'])
                    each having dict_keys(['text', 'label', 'combo'])
                    each having list of len num_test_samples each being tuple of (test_sample text, closest train sample text)
        """
        if type(column) != list:
            column = [column]
        result_score = {}
        result_detail = {}
        for k, t in self._ngrams.items():
            if use_cosine == 'bow':
                comparer = OverlapDetector(CosineSimilarityComparer(t), ssvise)
            elif use_cosine == 'f1':
                comparer = OverlapDetector(F1SimilarityComparer(), ssvise)
            elif use_cosine == 'emb':    
                comparer = OverlapDetectorEmbedding(CosineSimilarityEmbeddingComparer(), train_emb, test_emb, ssvise)

            comparison_result = comparer.compare(test, train, columns=column, 
                                                 add_combo=add_combo, answer_thresh=answer_thresh, answer_col=answer_col, 
                                                 question_thresh=question_thresh)
            result_score[k] = {}
            result_detail[k] = {}               
            for c in comparison_result:
                scores = comparison_result[c]["score"]
                detail = comparison_result[c]["details"]
                result_score[k][c] = scores
                result_detail[k][c] = detail
                
        return result_score, result_detail

    def print_summary(self, result_score, result_detail, ngrams=['Unigram'], columns=['combo']):
        for k in result_score:
            for c, scores in result_score[k].items():
                if (k in ngrams) and (c in columns):
                    score_stats = {"type": k, "column": c , "min": np.min(scores), "max": np.max(scores), "std": np.std(scores),
                                   "mean": np.mean(scores),
                                   "median": np.median(scores)}
                    print(json.dumps(score_stats, indent=1))

        for k in result_detail:
            for c, v in result_detail[k].items():
                if (k in ngrams) and (c in columns):
                    top_k = np.argsort(result_score[k][c])[-5:]
                    top_scores = np.array(result_score[k][c])[top_k]
                    print(k, c, top_scores.round(2).tolist())
                    print(json.dumps(np.array(v)[top_k].tolist(), indent=1))
        return


class SimilarityAggregator:
    """ Given a set of trainset:testset max similarity assignments, calculate various summaries 
        over most similar test example to each train example over all training datasets. 
        Then run reports against these
        
        usage: run the run_all_reports(...) function to instantiate instances of this class and run reports
    """
    def __init__(self, sim_results, no_overlap_thresh=40.0, 
                 results_list=[], compare_over='ALL', 
                 thresh_buckets=[0,60,90,101], logdir=''):
        """ 
        In general, initialising an instance of this class will take a similarity file and a set of results files
        as inputs and calculate a set of summary dictionaries that are then used to generate various reports from.
        Dictionary formats:

        initial similarity file format:
            sim_results[trainset][testset] = {'sim_scores': result_score,   # format described in SimilarityEvaluator ['Unigram']['column'] = [num_testset sized list of sim scores - each is highest sim score for each test example against this trainset examples]
                                              'sim_details': result_detail, # format described in SimilarityEvaluator ['Unigram']['column'] = [num_testset sized list of most similar example from this trainset ]
                                              'test_metric': prefmetric,    # metric type of test_scores
                                              'test_scores': scores}        # list of individual sample pred scores for this test set of type test_metric (now ignored since we now read/report the prediction scores of multiple different experiments)

        reversed version: (for each test sample what is the most similar train example in EACH traning dataset)
            sim_results_rev[testset][trainset] = {'sim_scores': result_score,   # format described in SimilarityEvaluator 
                                              'sim_details': result_detail, # format described in SimilarityEvaluator
                                              'test_metric': prefmetric,    # metric type of test_scores
                                              'test_scores': scores}        # list of individual sample scores for this test set

        summary version: (for each test sample what is the most similar train example overall)        
            sim_results_max[testset]['max_sim_over_train'] =
                                            {'sim_scores': {},    # format described in SimilarityEvaluator 
                                             'sim_details': {},   # format described in SimilarityEvaluator 
                                             'sim_train_ds': {},  # the training dataset the most similar example is from
                                             'test_metric': prefmetric,    # metric type of test_scores
                                             'test_scores': scores}        # list of individual sample scores for this test set from the eval_metrics.json file in the output dir (overridden below by a list of scores from multiple output directories)
                                             }
        similarity combined with pred performance:
            sim_results_summary[testset][ngram=Unigram][column=text/label/combo][trainset] =
                                             {'num_similar': 0,
                                             'percent_similar': 0.0,
                                             'mean_sim_score': 0.0,
                                             'min_sim_score': 0.0,
                                             'max_sim_score': 0.0,
                                             'median_sim_score': 0.0,
                                             'std_sim_score': 0.0,
                                             'comp3runs046__mean_pred_score': 0.0,
                                             'comp3runs046__min_pred_score': 0.0,
                                             'comp3runs046__max_pred_score': 0.0,
                                             'comp3runs046__median_pred_score': 0.0,
                                             'comp3runs046__std_pred_score': 0.0,
                                             'comp3runs046__most_similar_example': ('?', '?')}

        similarity combined with pred performance further broken down by similarity category:
            sim_results_summary_thresh[testset][ngram=Unigram][column=text/label/combo][trainset][similarity bucket] =
                                             {'num_similar': 0,
                                             'percent_similar': 0.0,
                                             'mean_sim_score': 0.0,
                                             'min_sim_score': 0.0,
                                             'max_sim_score': 0.0,
                                             'median_sim_score': 0.0,
                                             'std_sim_score': 0.0,
                                             'comp3runs046__mean_pred_score': 0.0,
                                             'comp3runs046__min_pred_score': 0.0,
                                             'comp3runs046__max_pred_score': 0.0,
                                             'comp3runs046__median_pred_score': 0.0,
                                             'comp3runs046__std_pred_score': 0.0,
                                             'comp3runs046__most_similar_example': ('?', '?')}

            eval_results[testset][shortname_of_outputdir] =
                                            {'test_metric': prefmetric,
                                             'test_scores': curr_scores, # not * 100
                                             'test_score': curr_score,    # already * 100
                                             'predictions': curr_preds  # list of str preds
                                             }

        """
        self.logdir = logdir
        self.compare_over = compare_over
        if type(self.compare_over) == str:
            self.compare_over = self.compare_over.upper()  #'ALL' means analyse over all train datasets. 'UQA' means original UnifiedQA train datasets only
        if self.compare_over == 'ALL':
            self.compare_over = ['ALL']
        elif self.compare_over == 'UQA':
            self.compare_over = eval_metrics.unifiedqa_base_train
        else:
            assert type(self.compare_over) == list, f"Error: compare_over param must be 'ALL', 'UQA' or list of training datasets to compare over not {self.compare_over}."
        self.thresh_buckets = thresh_buckets
        self.sim_results = sim_results
        self.sim_results_rev = {}
        self.sim_results_max = {}
        self.sim_results_summary = {}
        self.sim_results_summary_thresh = {}
        self.eval_results = {}
        self.outlist = []   #list of comma delimited output strings used in crosstab functions to create the output files. Saved here so that different output files can be combined to a new output eliminating the need to manual cut and paste in Excel
        print("Reversing train - test keys...")
        self.reverse_train_test()
        print("Calculating max similarity for each test set example over all train set examples...")
        self.max_over_trainsets()
        if len(results_list) > 0:
            print("Loading scores from eval_metrics.json files...")
            self.load_eval_results(results_list)
            print(f"Calculating extended summary matrix with no_overlap threshold: {no_overlap_thresh}...")
            self.summary_matrix_ext(no_overlap_thresh=no_overlap_thresh)
            print(f"Calculating summary matrix with similarity buckets with no_overlap threshold: {no_overlap_thresh}...")
            self.summary_matrix_extbythresh(no_overlap_thresh=no_overlap_thresh)
        else:  #run calculations using the single default prediction set already included in sim_results
            print(f"Calculating summary matrix with no_overlap threshold: {no_overlap_thresh}...")
            self.summary_matrix(no_overlap_thresh=no_overlap_thresh)
        print("Finished calculating summary matrix!")
        return
    
    def reverse_train_test(self):
        """ Reverse order of train and test keys in dict only including train datasets in self.compare_over
        """
        for trainset in self.sim_results:
            if (self.compare_over == ['ALL']) or (trainset in self.compare_over):
                for testset in self.sim_results[trainset]:
                    if self.sim_results_rev.get(testset) is None:
                        self.sim_results_rev[testset] = {}
                    self.sim_results_rev[testset][trainset] = self.sim_results[trainset][testset]
        return

    def max_over_trainsets(self):
        """ Calc single most similar example over all training sets for each test set example
        """
        for testset in self.sim_results_rev:
            if self.sim_results_max.get(testset) is None:
                self.sim_results_max[testset] = {}
                self.sim_results_max[testset]['max_sim_over_train'] = {'sim_scores': {}, 'sim_details': {}, 'sim_train_ds': {}}
                first_trainset = list(self.sim_results_rev[testset].keys())[0]
                self.sim_results_max[testset]['max_sim_over_train']['test_metric'] = self.sim_results_rev[testset][first_trainset]['test_metric']
                self.sim_results_max[testset]['max_sim_over_train']['test_scores'] = self.sim_results_rev[testset][first_trainset]['test_scores']

            for trainset in self.sim_results_rev[testset]:
                curr_result = self.sim_results_rev[testset][trainset]

                for ngram in curr_result['sim_scores']:                        
                    if self.sim_results_max[testset]['max_sim_over_train']['sim_scores'].get(ngram) is None:
                        self.sim_results_max[testset]['max_sim_over_train']['sim_scores'][ngram] = {}    
                        self.sim_results_max[testset]['max_sim_over_train']['sim_details'][ngram] = {}    
                        self.sim_results_max[testset]['max_sim_over_train']['sim_train_ds'][ngram] = {}    

                    for column in curr_result['sim_scores'][ngram]:
                        if self.sim_results_max[testset]['max_sim_over_train']['sim_scores'][ngram].get(column) is None:
                            self.sim_results_max[testset]['max_sim_over_train']['sim_scores'][ngram][column] = [-1.0] * len(self.sim_results_rev[testset][trainset]['sim_scores'][ngram][column])
                            self.sim_results_max[testset]['max_sim_over_train']['sim_details'][ngram][column] = [ ('?','?') ] * len(self.sim_results_rev[testset][trainset]['sim_scores'][ngram][column])
                            self.sim_results_max[testset]['max_sim_over_train']['sim_train_ds'][ngram][column] = ['?'] * len(self.sim_results_rev[testset][trainset]['sim_scores'][ngram][column])

                        for i, simscore in enumerate(curr_result['sim_scores'][ngram][column]):
                            if self.sim_results_max[testset]['max_sim_over_train']['sim_scores'][ngram][column][i] < simscore:
                                self.sim_results_max[testset]['max_sim_over_train']['sim_scores'][ngram][column][i] = simscore
                                self.sim_results_max[testset]['max_sim_over_train']['sim_details'][ngram][column][i] = curr_result['sim_details'][ngram][column][i]
                                self.sim_results_max[testset]['max_sim_over_train']['sim_train_ds'][ngram][column][i] = trainset
        return

    def load_eval_results(self, results_list):
        """ Load eval results from 1+ eval_metrics.json files...
        """
        res = eval_metrics.OutputResults(results_list)
        for i, result_file in enumerate(res.dataset_metrics):
            curr_shortname = res.shortnames[i]
            curr_outmetrics = res.dataset_metrics[result_file]
            for testset in curr_outmetrics.results_dict.keys():
                prefmetric = curr_outmetrics.get_pref_metric(testset)
                if prefmetric == 'RL':  # No sample-level results for RL
                    prefmetric = 'F1'
                curr_scores = curr_outmetrics.results_dict[testset][prefmetric]['scores']
                curr_score = curr_outmetrics.get_value(testset, prefmetric, 'score')
                curr_preds = curr_outmetrics.results_dict[testset]['predictions']
                if self.eval_results.get(testset) is None:
                    self.eval_results[testset] = {}
                if self.eval_results[testset].get(curr_shortname) is None:
                    self.eval_results[testset][curr_shortname] = {}
                self.eval_results[testset][curr_shortname] = {'test_metric': prefmetric,
                                                              'test_scores': curr_scores,
                                                              'test_score': curr_score,
                                                              'predictions': curr_preds
                                                              }
        return

    def summary_matrix(self, no_overlap_thresh=1.0):
        """ Create summary dict to make it easy to output tables of test sets on y axis and train sets on x axis.
            Creates a special "no significant overlap with any trainset" column for test samples with similarity scores < no_overlap_thresh
        """
        trainsets = ['no_overlap'] + list(self.sim_results.keys())
        template = {'num_similar': 0, 
                    'percent_similar': 0.0, 
                    'mean_sim_score': 0.0,
                    'min_sim_score': 0.0,
                    'max_sim_score': 0.0,
                    'median_sim_score': 0.0,
                    'std_sim_score' : 0.0,
                    'mean_pred_score': 0.0,
                    'min_pred_score': 0.0,
                    'max_pred_score': 0.0,
                    'median_pred_score': 0.0,
                    'std_pred_score': 0.0,
                    'mean_pred_score_str': '',
                    'most_similar_example': ('?', '?') }
        
        for testset in self.sim_results_max:
            print(f"Calculating for test set: {testset}..")
            self.sim_results_summary[testset] = {'test_metric': self.sim_results_max[testset]['max_sim_over_train']['test_metric']}
            pred_np = np.array(self.sim_results_max[testset]['max_sim_over_train']['test_scores'])

            for ngram in self.sim_results_max[testset]['max_sim_over_train']['sim_scores']:
                if self.sim_results_summary[testset].get(ngram) is None:
                    self.sim_results_summary[testset][ngram] = {}
                
                for column in self.sim_results_max[testset]['max_sim_over_train']['sim_scores'][ngram]:
                    if self.sim_results_summary[testset][ngram].get(column) is None:
                        self.sim_results_summary[testset][ngram][column] = {} 
                    simscores_np = np.array(self.sim_results_max[testset]['max_sim_over_train']['sim_scores'][ngram][column])
                    simexamples_np = np.array(self.sim_results_max[testset]['max_sim_over_train']['sim_details'][ngram][column])
                    train_np = np.array(self.sim_results_max[testset]['max_sim_over_train']['sim_train_ds'][ngram][column], dtype=object)
                    indices_underthresh = np.where(simscores_np < no_overlap_thresh)
                    train_np[indices_underthresh] = 'no_overlap'
                    for trainset in trainsets:
                        if self.sim_results_summary[testset][ngram][column].get(trainset) is None:
                            self.sim_results_summary[testset][ngram][column][trainset] = copy.deepcopy(template)
                        indices_trainset = np.where(train_np == trainset)
                        num_selected = indices_trainset[0].shape[0]
                        if num_selected > 0:
                            self.sim_results_summary[testset][ngram][column][trainset]['num_similar'] = num_selected
                            self.sim_results_summary[testset][ngram][column][trainset]['percent_similar'] = num_selected / len(simscores_np)

                            self.sim_results_summary[testset][ngram][column][trainset]['mean_sim_score'] = float(np.mean(simscores_np[indices_trainset]))
                            self.sim_results_summary[testset][ngram][column][trainset]['min_sim_score'] = float(np.min(simscores_np[indices_trainset]))
                            self.sim_results_summary[testset][ngram][column][trainset]['max_sim_score'] = float(np.max(simscores_np[indices_trainset]))
                            self.sim_results_summary[testset][ngram][column][trainset]['median_sim_score'] = float(np.median(simscores_np[indices_trainset]))
                            self.sim_results_summary[testset][ngram][column][trainset]['std_sim_score'] = float(np.std(simscores_np[indices_trainset]))

                            self.sim_results_summary[testset][ngram][column][trainset]['mean_pred_score'] = float(np.mean(pred_np[indices_trainset])) * 100.0
                            self.sim_results_summary[testset][ngram][column][trainset]['min_pred_score'] = float(np.min(pred_np[indices_trainset])) * 100.0
                            self.sim_results_summary[testset][ngram][column][trainset]['max_pred_score'] = float(np.max(pred_np[indices_trainset])) * 100.0
                            self.sim_results_summary[testset][ngram][column][trainset]['median_pred_score'] = float(np.median(pred_np[indices_trainset])) * 100.0
                            self.sim_results_summary[testset][ngram][column][trainset]['std_pred_score'] = float(np.std(pred_np[indices_trainset])) * 100.0
                            self.sim_results_summary[testset][ngram][column][trainset]['mean_pred_score_str'] = str(round(float(np.mean(pred_np[indices_trainset])) * 100.0, 2)) + ' (' + str(num_selected) + ')'
                            
                            best_idx = np.argmax(simscores_np[indices_trainset])
                            self.sim_results_summary[testset][ngram][column][trainset]['most_similar_example'] = tuple(simexamples_np[indices_trainset][best_idx])
        return

    def summary_matrix_ext(self, no_overlap_thresh=1.0):
        """ Create summary dict to make it easy to output tables of test sets on y axis and train sets on x axis.
            Creates a special "no significant overlap with any trainset" column for test samples with similarity scores < no_overlap_thresh
            
            This extended version adds prediction stats for a set of output predictions not just for the default one...
        """
        trainsets = ['no_overlap'] + list(self.sim_results.keys())
        first_result = list(self.eval_results.keys())[0]
        results_list = list(self.eval_results[first_result].keys())
        template = {'num_similar': 0, 
                    'percent_similar': 0.0, 
                    'mean_sim_score': 0.0,
                    'min_sim_score': 0.0,
                    'max_sim_score': 0.0,
                    'median_sim_score': 0.0,
                    'std_sim_score' : 0.0}
        for result in results_list:
            template[result + '__mean_pred_score'] = 0.0
            template[result + '__min_pred_score'] = 0.0
            template[result + '__max_pred_score'] = 0.0
            template[result + '__median_pred_score'] = 0.0
            template[result + '__std_pred_score'] = 0.0
            template[result + '__mean_pred_score_str'] = ''
            template[result + '__most_similar_example'] = ('?', '?')
        
        for testset in self.sim_results_max:
            print(f"Calculating for test set: {testset}..")
            self.sim_results_summary[testset] = {'test_metric': self.sim_results_max[testset]['max_sim_over_train']['test_metric']}

            for ngram in self.sim_results_max[testset]['max_sim_over_train']['sim_scores']:
                if self.sim_results_summary[testset].get(ngram) is None:
                    self.sim_results_summary[testset][ngram] = {}
                
                for column in self.sim_results_max[testset]['max_sim_over_train']['sim_scores'][ngram]:
                    if self.sim_results_summary[testset][ngram].get(column) is None:
                        self.sim_results_summary[testset][ngram][column] = {} 
                    simscores_np = np.array(self.sim_results_max[testset]['max_sim_over_train']['sim_scores'][ngram][column])
                    simexamples_np = np.array(self.sim_results_max[testset]['max_sim_over_train']['sim_details'][ngram][column])
                    train_np = np.array(self.sim_results_max[testset]['max_sim_over_train']['sim_train_ds'][ngram][column], dtype=object)
                    indices_underthresh = np.where(simscores_np < no_overlap_thresh)
                    train_np[indices_underthresh] = 'no_overlap'
                    for trainset in trainsets:
                        if self.sim_results_summary[testset][ngram][column].get(trainset) is None:
                            self.sim_results_summary[testset][ngram][column][trainset] = copy.deepcopy(template)
                        indices_trainset = np.where(train_np == trainset)
                        num_selected = indices_trainset[0].shape[0]
                        if num_selected > 0:
                            self.sim_results_summary[testset][ngram][column][trainset]['num_similar'] = num_selected
                            self.sim_results_summary[testset][ngram][column][trainset]['percent_similar'] = num_selected / len(simscores_np)

                            self.sim_results_summary[testset][ngram][column][trainset]['mean_sim_score'] = float(np.mean(simscores_np[indices_trainset]))
                            self.sim_results_summary[testset][ngram][column][trainset]['min_sim_score'] = float(np.min(simscores_np[indices_trainset]))
                            self.sim_results_summary[testset][ngram][column][trainset]['max_sim_score'] = float(np.max(simscores_np[indices_trainset]))
                            self.sim_results_summary[testset][ngram][column][trainset]['median_sim_score'] = float(np.median(simscores_np[indices_trainset]))
                            self.sim_results_summary[testset][ngram][column][trainset]['std_sim_score'] = float(np.std(simscores_np[indices_trainset]))
                            
                            for result in results_list:
                                pred_np = np.array(self.eval_results[testset][result]['test_scores'])
                                self.sim_results_summary[testset][ngram][column][trainset][result + '__mean_pred_score'] = float(np.mean(pred_np[indices_trainset])) * 100.0
                                self.sim_results_summary[testset][ngram][column][trainset][result + '__min_pred_score'] = float(np.min(pred_np[indices_trainset])) * 100.0
                                self.sim_results_summary[testset][ngram][column][trainset][result + '__max_pred_score'] = float(np.max(pred_np[indices_trainset])) * 100.0
                                self.sim_results_summary[testset][ngram][column][trainset][result + '__median_pred_score'] = float(np.median(pred_np[indices_trainset])) * 100.0
                                self.sim_results_summary[testset][ngram][column][trainset][result + '__std_pred_score'] = float(np.std(pred_np[indices_trainset])) * 100.0
                                self.sim_results_summary[testset][ngram][column][trainset][result + '__mean_pred_score_str'] = str(round(float(np.mean(pred_np[indices_trainset])) * 100.0, 2)) + ' (' + str(num_selected) + ')'
                                
                                best_idx = np.argmax(simscores_np[indices_trainset])
                                self.sim_results_summary[testset][ngram][column][trainset][result + '__most_similar_example'] = tuple(simexamples_np[indices_trainset][best_idx])
        return


    def summary_matrix_extbythresh(self, no_overlap_thresh=1.0):
        """ Create summary dict to make it easy to output tables of test sets on y axis and train sets on x axis.
            Creates a special "no significant overlap with any trainset" column for test samples with similarity scores < no_overlap_thresh
            
            This version adds prediction stats split into threshold buckets...
        """
        trainsets = ['no_overlap'] + list(self.sim_results.keys())
        first_result = list(self.eval_results.keys())[0]
        results_list = list(self.eval_results[first_result].keys())
        template = {'num_similar': 0, 
                    'percent_similar': 0.0,
                    'mean_sim_score': 0.0,
                    'min_sim_score': 0.0,
                    'max_sim_score': 0.0,
                    'median_sim_score': 0.0,
                    'std_sim_score' : 0.0}
        for result in results_list:          
            template[result + '__mean_pred_score'] = 0.0
            template[result + '__min_pred_score'] = 0.0
            template[result + '__max_pred_score'] = 0.0
            template[result + '__median_pred_score'] = 0.0
            template[result + '__std_pred_score'] = 0.0
            template[result + '__mean_pred_score_str'] = ''
            template[result + '__most_similar_example'] = ('?', '?')
        
        for testset in self.sim_results_max:
            print(f"Calculating for test set: {testset}..")
            self.sim_results_summary_thresh[testset] = {'test_metric': self.sim_results_max[testset]['max_sim_over_train']['test_metric']}

            for ngram in self.sim_results_max[testset]['max_sim_over_train']['sim_scores']:
                if self.sim_results_summary_thresh[testset].get(ngram) is None:
                    self.sim_results_summary_thresh[testset][ngram] = {}
                
                for column in self.sim_results_max[testset]['max_sim_over_train']['sim_scores'][ngram]:
                    if self.sim_results_summary_thresh[testset][ngram].get(column) is None:
                        self.sim_results_summary_thresh[testset][ngram][column] = {} 
                    simscores_np = np.array(self.sim_results_max[testset]['max_sim_over_train']['sim_scores'][ngram][column])
                    simexamples_np = np.array(self.sim_results_max[testset]['max_sim_over_train']['sim_details'][ngram][column])
                    train_np = np.array(self.sim_results_max[testset]['max_sim_over_train']['sim_train_ds'][ngram][column], dtype=object)
                    indices_underthresh = np.where(simscores_np < no_overlap_thresh)
                    train_np[indices_underthresh] = 'no_overlap'
                    for trainset in trainsets:
                        if self.sim_results_summary_thresh[testset][ngram][column].get(trainset) is None:
                            self.sim_results_summary_thresh[testset][ngram][column][trainset] = {}

                        bucket_bottom = self.thresh_buckets[0]    
                        for bucket_top in self.thresh_buckets[1:]:
                            bucketstr = str(bucket_bottom) + ':' + str(bucket_top)
                            if self.sim_results_summary_thresh[testset][ngram][column][trainset].get(bucketstr) is None:
                                self.sim_results_summary_thresh[testset][ngram][column][trainset][bucketstr] = copy.deepcopy(template)

                            indices_trainset = np.where(train_np == trainset)
                            num_selected = indices_trainset[0].shape[0]
                            if num_selected > 0:
                                simscores_train = simscores_np[indices_trainset]
                                indices_bucket = np.where((simscores_train >= bucket_bottom) & (simscores_train < bucket_top))
                                num_selected = indices_bucket[0].shape[0]
                                if num_selected > 0:
                                    simexamples_train = simexamples_np[indices_trainset]
                                    
                                    self.sim_results_summary_thresh[testset][ngram][column][trainset][bucketstr]['num_similar'] = num_selected
                                    self.sim_results_summary_thresh[testset][ngram][column][trainset][bucketstr]['percent_similar'] = num_selected / len(simscores_train)
        
                                    self.sim_results_summary_thresh[testset][ngram][column][trainset][bucketstr]['mean_sim_score'] = float(np.mean(simscores_train[indices_bucket]))
                                    self.sim_results_summary_thresh[testset][ngram][column][trainset][bucketstr]['min_sim_score'] = float(np.min(simscores_train[indices_bucket]))
                                    self.sim_results_summary_thresh[testset][ngram][column][trainset][bucketstr]['max_sim_score'] = float(np.max(simscores_train[indices_bucket]))
                                    self.sim_results_summary_thresh[testset][ngram][column][trainset][bucketstr]['median_sim_score'] = float(np.median(simscores_train[indices_bucket]))
                                    self.sim_results_summary_thresh[testset][ngram][column][trainset][bucketstr]['std_sim_score'] = float(np.std(simscores_train[indices_bucket]))

                                    # Added to faciliate dump of most similar in each bucket:
                                    self.sim_results_summary_thresh[testset][ngram][column][trainset][bucketstr]['sim_scores'] = simscores_train[indices_bucket]
                                    self.sim_results_summary_thresh[testset][ngram][column][trainset][bucketstr]['sim_details'] = simexamples_train[indices_bucket]
                                    
                                    for result in results_list:
                                        pred_np = np.array(self.eval_results[testset][result]['test_scores'])
                                        pred_np = pred_np[indices_trainset]
                                        self.sim_results_summary_thresh[testset][ngram][column][trainset][bucketstr][result + '__mean_pred_score'] = float(np.mean(pred_np[indices_bucket])) * 100.0
                                        self.sim_results_summary_thresh[testset][ngram][column][trainset][bucketstr][result + '__min_pred_score'] = float(np.min(pred_np[indices_bucket])) * 100.0
                                        self.sim_results_summary_thresh[testset][ngram][column][trainset][bucketstr][result + '__max_pred_score'] = float(np.max(pred_np[indices_bucket])) * 100.0
                                        self.sim_results_summary_thresh[testset][ngram][column][trainset][bucketstr][result + '__median_pred_score'] = float(np.median(pred_np[indices_bucket])) * 100.0
                                        self.sim_results_summary_thresh[testset][ngram][column][trainset][bucketstr][result + '__std_pred_score'] = float(np.std(pred_np[indices_bucket])) * 100.0
                                        self.sim_results_summary_thresh[testset][ngram][column][trainset][bucketstr][result + '__mean_pred_score_str'] = str(round(float(np.mean(pred_np[indices_bucket])) * 100.0, 2)) + ' (' + str(num_selected) + ')'
                                        
                                        best_idx = np.argmax(simscores_train[indices_bucket])
                                        self.sim_results_summary_thresh[testset][ngram][column][trainset][bucketstr][result + '__most_similar_example'] = tuple(simexamples_train[indices_bucket][best_idx])
                            bucket_bottom = bucket_top
        return


    
    def crosstab_x_train_y_eval(self, dsetset='ALL', output_metrics = 'ALL', ngram='Unigram', column='combo', outname = 'tmp_crosstab.txt'):
        """ Output crosstab with x axis being train datasets and y axis being eval datasets 
        """
        outfile = os.path.join(self.logdir, outname)
        outlist = []
        if type(dsetset) == str and dsetset.upper() == 'ALL':
            dsetset = list(self.sim_results_summary.keys())
        elif type(dsetset) == str:
            dsetset = [dsetset]
            
        if type(output_metrics) == str and output_metrics.upper() == 'ALL':
            output_metrics = ['num_similar', 'percent_similar', 
                              'mean_sim_score', 'min_sim_score', 'max_sim_score', 'median_sim_score', 'std_sim_score', 
                              'mean_pred_score', 'min_pred_score', 'max_pred_score', 'median_pred_score', 'std_pred_score', 'mean_pred_score_str',
                              'most_similar_example']           
        elif type(output_metrics) == str:
            output_metrics = [output_metrics]
            
        trainsets = ['no_overlap'] + list(self.sim_results.keys())
            
        header = 'Eval Dataset,Metric'        
        for trainset in trainsets:
            header = header + ',' + trainset
        print(header)  
        outlist.append(header)

        for dset in dsetset:
            for i, m in enumerate(output_metrics):
                if i == 0:
                    col_start = dset + ','
                else:
                    col_start = ','
                col_metric = m
                if m in ['mean_pred_score', 'min_pred_score', 'max_pred_score', 'median_pred_score', 'std_pred_score', 'mean_pred_score_str']:
                    col_metric = col_metric + ' (' + self.sim_results_summary[dset]['test_metric'] + ')'
                outstr = col_start + col_metric
                for j, trainset in enumerate(trainsets):
                    outstr += ','
                    if m == 'most_similar_example':
                        if self.sim_results_summary[dset][ngram][column][trainset]['most_similar_example'][0] != '?':
                            outstr += "TEST: " + self.sim_results_summary[dset][ngram][column][trainset]['most_similar_example'][0].replace(',', '') + \
                                     "  TRAIN: " + self.sim_results_summary[dset][ngram][column][trainset]['most_similar_example'][1].replace(',', '')
                    else:
                        outstr += str(self.sim_results_summary[dset][ngram][column][trainset][m])
                print(outstr)
                outlist.append(outstr)
                
        with open(outfile, 'w') as f:
            f.write('\r\n'.join(outlist))
        self.outlist = outlist
        return


    def crosstab_x_train_y_eval_ext(self, dsetset='ALL', output_metrics = 'ALL', output_results = 'ALL',
                                    ngram='Unigram', column='combo', outname = 'tmp_crosstab.txt'):
        """ Output crosstab with x axis being train datasets and y axis being eval datasets 
            Extended to output a line for each selected result set
        """
        outfile = os.path.join(self.logdir, outname)
        outlist = []
        if type(dsetset) == str and dsetset.upper() == 'ALL':
            dsetset = list(self.sim_results_summary.keys())
        elif type(dsetset) == str:
            dsetset = [dsetset]

        if type(output_results) == str and output_results.upper() == 'ALL':
            first_result = list(self.eval_results.keys())[0]
            output_results = list(self.eval_results[first_result].keys())
        elif type(output_results) == str:
            output_results = [output_results]
            
        if type(output_metrics) == str and output_metrics.upper() == 'ALL':
            output_metrics = ['num_similar', 'percent_similar', 
                              'mean_sim_score', 'min_sim_score', 'max_sim_score', 'median_sim_score', 'std_sim_score', 
                              'mean_pred_score', 'min_pred_score', 'max_pred_score', 'median_pred_score', 'std_pred_score', 'mean_pred_score_str',
                              'most_similar_example']
        elif type(output_metrics) == str:
            output_metrics = [output_metrics]

        include_key = []
        sim_template = {'num_similar', 'percent_similar', 'mean_sim_score', 'min_sim_score', 'max_sim_score', 'median_sim_score', 'std_sim_score'}
        for m in output_metrics:
            if m in sim_template:
                include_key.append(m)       # metrics not prepended by the model output name ie those related to test-train similarity
        pred_template = {'mean_pred_score', 'min_pred_score', 'max_pred_score', 'median_pred_score', 'std_pred_score', 'mean_pred_score_str', 'most_similar_example'}        
        for result in output_results:
            for m in output_metrics:
                if m in pred_template:
                    include_key.append(result + '__' + m)   # metrics prepended by the model output name ie those related to prediction accuracy
            
        trainsets = ['no_overlap'] + list(self.sim_results.keys())
            
        header = 'Eval Dataset,Metric'        
        for trainset in trainsets:
            header = header + ',' + trainset
        print(header)  
        outlist.append(header)

        for dset in dsetset:                    # for each eval dataset
            for i, m in enumerate(include_key):     # for each output metric
                if i == 0:
                    col_start = dset + ','
                else:
                    col_start = ','
                col_metric = m
                if m not in sim_template:   # if current metric prepended by model output name
                    sep_idx = m.find('__')
                    if sep_idx != -1:
                        coltype = m.split('__')[-1]
                        if coltype in ['mean_pred_score', 'min_pred_score', 'max_pred_score', 'median_pred_score', 'std_pred_score', 'mean_pred_score_str']:
                            res = m.split('__')[0]
                            col_metric = col_metric + ' (' + self.eval_results[dset][res]['test_metric'] + ': ' + str(round(self.eval_results[dset][res]['test_score'], 2)) + ')'
                outstr = col_start + col_metric
                for j, trainset in enumerate(trainsets):
                    outstr += ','
                    if m.endswith('most_similar_example'):
                        if self.sim_results_summary[dset][ngram][column][trainset][m][0] != '?':
                            outstr += "TEST: " + self.sim_results_summary[dset][ngram][column][trainset][m][0].replace(',', '') + \
                                     "  TRAIN: " + self.sim_results_summary[dset][ngram][column][trainset][m][1].replace(',', '')
                    else:
                        outstr += str(self.sim_results_summary[dset][ngram][column][trainset][m])
                print(outstr)
                outlist.append(outstr)
                
        with open(outfile, 'w') as f:
            f.write('\r\n'.join(outlist))            
        self.outlist = outlist
        return


    def crosstab_x_train_y_evalbythresh(self, dsetset='ALL', output_metrics = 'ALL', output_results = 'ALL',
                                    ngram='Unigram', column='combo', outname = 'tmp_crosstab.txt'):
        """ Output crosstab with x axis being train datasets 
            and y axis being eval datasets bucketted by similarity score
            (and Extended to output a line for each selected result set)
        """
        outfile = os.path.join(self.logdir, outname)
        outlist = []
        if type(dsetset) == str and dsetset.upper() == 'ALL':
            dsetset = list(self.sim_results_summary_thresh.keys())
        elif type(dsetset) == str:
            dsetset = [dsetset]
        first_result = dsetset[0]   # first testset
        first_train = list(self.sim_results_summary_thresh[first_result][ngram][column].keys())[0]  # first train set
        buckets = list(self.sim_results_summary_thresh[first_result][ngram][column][first_train].keys())  # list of similarity bucket keys eg ['-1000:10', '10:20', '20:30', '30:40', '40:50', '50:60', '60:70', '70:80', '80:90', '90:101']

        if type(output_results) == str and output_results.upper() == 'ALL':
            #first_result = list(self.eval_results.keys())[0]
            output_results = list(self.eval_results[first_result].keys())
        elif type(output_results) == str:
            output_results = [output_results]
            
        if type(output_metrics) == str and output_metrics.upper() == 'ALL':
            output_metrics = ['num_similar', 'percent_similar', 
                              'mean_sim_score', 'min_sim_score', 'max_sim_score', 'median_sim_score', 'std_sim_score', 
                              'mean_pred_score', 'min_pred_score', 'max_pred_score', 'median_pred_score', 'std_pred_score', 'mean_pred_score_str',
                              'most_similar_example']
        elif type(output_metrics) == str:
            output_metrics = [output_metrics]

        include_key = []
        sim_template = {'num_similar', 'percent_similar', 'mean_sim_score', 'min_sim_score', 'max_sim_score', 'median_sim_score', 'std_sim_score'}
        for m in output_metrics:
            if m in sim_template:
                include_key.append(m)  # metrics not prepended by the model output name ie those related to test-train similarity
        pred_template = {'mean_pred_score', 'min_pred_score', 'max_pred_score', 'median_pred_score', 'std_pred_score', 'mean_pred_score_str', 'most_similar_example'}
        for result in output_results:
            for m in output_metrics:
                if m in pred_template:
                    include_key.append(result + '__' + m)  # metrics prepended by the model output name ie those related to prediction accuracy
            
        trainsets = ['no_overlap'] + list(self.sim_results.keys())
        
        header = 'Eval Dataset,Similarity,Metric' 
        for trainset in trainsets:
            header = header + ',' + trainset
        print(header)
        outlist.append(header)

        for dset in dsetset:  # for each eval dataset
            for b, bucket in enumerate(buckets):  # for each similarity score bucket
                for i, m in enumerate(include_key):   # for each output metric
                    if b == 0 and i==0:
                        col_start = dset + ','
                    else:
                        col_start = ','
                    if i == 0:
                        col_start += '"' + bucket + '",'  # double quotes to prevent Excel misinterpreting on import..
                    else:
                        col_start += ','
                    col_metric = m
                    if m not in sim_template:  # if current metric prepended by model output name
                        sep_idx = m.find('__')
                        if sep_idx != -1:
                            coltype = m.split('__')[-1]
                            if coltype in ['mean_pred_score', 'min_pred_score', 'max_pred_score', 'median_pred_score', 'std_pred_score', 'mean_pred_score_str']:
                                res = m.split('__')[0]
                                col_metric = col_metric + ' (' + self.eval_results[dset][res]['test_metric'] + ': ' + str(round(self.eval_results[dset][res]['test_score'], 2)) + ')'
                    outstr = col_start + col_metric
                    for j, trainset in enumerate(trainsets):
                        outstr += ','
                        if m.endswith('most_similar_example'):
                            if self.sim_results_summary_thresh[dset][ngram][column][trainset][bucket][m][0] != '?':
                                outstr += "TEST: " + self.sim_results_summary_thresh[dset][ngram][column][trainset][bucket][m][0].replace(',', '') + \
                                         "  TRAIN: " + self.sim_results_summary_thresh[dset][ngram][column][trainset][bucket][m][1].replace(',', '')
                        else:
                            outstr += str(self.sim_results_summary_thresh[dset][ngram][column][trainset][bucket][m])
                    print(outstr)
                    outlist.append(outstr)
                
        with open(outfile, 'w') as f:
            f.write('\r\n'.join(outlist))
        self.outlist = outlist
        return


def output_interleave(outlist_a, outlist_b, 
                      b_add = ' +tdnd', outname='tmp_interleave.txt',
                      choose_bucket = 'ALL'):
    """ Take two output lists of csv strings of same length, interleave them and output the resulting file.
    Usage:
        new_outlist = output_interleave(s_uqa.outlist, s_tdnd.outlist, choose_bucket = ['90:101'])
        or eg new_outlist = output_interleave(s_uqa.outlist, s_tdnd.outlist, choose_bucket = ['60:90', '90:101'])
    """
    if type(choose_bucket) != list:
        choose_bucket = [choose_bucket]
    new_outlist = []
    curr_ds = ""
    num_lines = len(outlist_a)
    for i in range(num_lines):
        if i == 0:  # no need to repeat column headers
            new_outlist.append(outlist_a[i])
        else:
            new_line_a = outlist_a[i].split(',')
            if new_line_a[0] != '':    # eval dataset name not repeated on each line so must remember it
                curr_ds = new_line_a[0]
            else:
                new_line_a[0] = curr_ds
            bucketstr = new_line_a[1].strip('"')
            if choose_bucket == ['ALL'] or bucketstr in choose_bucket:
                new_outlist.append( ','.join(new_line_a) )
                new_line_b = outlist_b[i].split(',')
                new_line_b[0] = curr_ds + b_add
                new_outlist.append( ','.join(new_line_b) )
    with open(outname, 'w') as f:
        f.write('\r\n'.join(new_outlist))
    return new_outlist


def calc_pred_difference(s_a, s_b, dsetset='ALL', ngram='Unigram', column='combo', 
                         outname = 'tmp_pred_diff.txt', extended=True, lowest_sim_bucket_only=False):
    """ s_a and s_b are two SimilarityAggregator objects, eg one with uqa only outputs and one with uqa + tdnd outputs
        For each eval dataset in dsetset identifies samples that "moved" to a different "most similar" trainset bucket
        (possibly also to different similarity bucket) and those that "stayed" with same "most similar" trainset.        
        Note: predictions are stored in eg s_tdnd.eval_results['testset']['outputdirshortname']. In this routine 
              in case of multiple prediction files here we take the first one
    """
    outfile = os.path.join(s_a.logdir, outname)
    if type(dsetset) != list:
        if dsetset.upper() == 'ALL':
            dsetset = list(s_a.sim_results_max.keys())
        else:
            dsetset = [dsetset]

    firsttime = True
    outlist = []
    outlist.append('Eval Dataset,Overall Mean Improvement,Stay Mean Improvement,Move Mean Improvement')
    for dset in dsetset:     # for each eval dataset

        sim_a = s_a.sim_results_max[dset]['max_sim_over_train']  # keys(['sim_scores', 'sim_details', 'sim_train_ds', 'test_metric', 'test_scores'])
        sim_b = s_b.sim_results_max[dset]['max_sim_over_train']  # keys(['sim_scores', 'sim_details', 'sim_train_ds', 'test_metric', 'test_scores'])

        sim_train_ds_a = np.array(sim_a['sim_train_ds'][ngram][column], dtype=object)
        sim_train_ds_b = np.array(sim_b['sim_train_ds'][ngram][column], dtype=object)
        
        if lowest_sim_bucket_only:
            top_of_lowest_bucket = s_a.thresh_buckets[1]
            sim_scores_a = np.array(sim_a['sim_scores'][ngram][column])
            lowest_bucket_indices = np.where(sim_scores_a < top_of_lowest_bucket)
            sim_train_ds_a = sim_train_ds_a[lowest_bucket_indices]
            sim_train_ds_b = sim_train_ds_b[lowest_bucket_indices]  
        
        first = list(s_a.eval_results[dset].keys())[0]      # get first set of eval results in case of having multiple
        preds_a = s_a.eval_results[dset][first]             # keys(['test_metric', 'test_scores', 'test_score'])
        metric = preds_a['test_metric']
        score_a = preds_a['test_score']
        scores_a = np.array(preds_a['test_scores']).astype(np.float32)

        first = list(s_b.eval_results[dset].keys())[0]
        preds_b = s_b.eval_results[dset][first]
        score_b = preds_b['test_score']
        scores_b = np.array(preds_b['test_scores']).astype(np.float32)

        if lowest_sim_bucket_only:
            scores_a = scores_a[lowest_bucket_indices]
            scores_b = scores_b[lowest_bucket_indices]
            score_a = np.mean(scores_a) * 100.0
            score_b = np.mean(scores_b) * 100.0

        score_diff = score_b - score_a
        scores_diff = scores_b - scores_a
        
        moved = np.where(sim_train_ds_a != sim_train_ds_b)
        stay = np.where(sim_train_ds_a == sim_train_ds_b)
        move_count = moved[0].shape[0]
        stay_count = stay[0].shape[0]
        total_count = scores_diff.shape[0]
        if move_count > 0:
            mean_improvement_moved = np.mean(scores_diff[moved]) * 100
        else:
            mean_improvement_moved = 0.0
        if stay_count > 0:
            mean_improvement_stay = np.mean(scores_diff[stay]) * 100
        else:
            mean_improvement_stay = 0.0
        outstr = f'{dset} ({metric}: {score_a:.2f}->{score_b:.2f}),{score_diff:.2f} ({total_count}),{mean_improvement_stay:.2f} ({stay_count}),{mean_improvement_moved:.2f} ({move_count})'
        
        if extended:
            moved_to = list(np.unique(sim_train_ds_b[moved]))
            if firsttime:
                firsttime = False
                for trainset in moved_to:
                    outlist[0] += ',' + trainset  # add column to header row

            for trainset in moved_to:
                moved_to_trainset = np.where(sim_train_ds_b == trainset)
                move_to_trainset_count = moved_to_trainset[0].shape[0]
                if move_to_trainset_count > 0:
                    mean_improvement_trainset = np.mean(scores_diff[moved_to_trainset]) * 100
                else:
                    mean_improvement_trainset = 0.0
                outstr += f',{mean_improvement_trainset:.2f} ({move_to_trainset_count})'

        outlist.append(outstr)
    with open(outfile, 'w') as f:
        f.write('\r\n'.join(outlist))
    return outlist


def output_most_similar(s, dsetset='ALL', buckets_select='ALL', ngram='Unigram', column='combo', 
                         outname = 'tmp_most_similar_dump.txt', topk=3):
    """ Basic not very pretty dump of most similar eval vs train pairs
    Usage: 
        new_outlist = output_most_similar(s_uqa, dsetset='ALL', buckets_select='ALL', ngram='Unigram', column='combo', topk=3)
        new_outlist = output_most_similar(s_tdnd, dsetset='ALL', buckets_select='ALL', ngram='Unigram', column='combo', topk=3)

        new_outlist = output_most_similar(s_uqa, dsetset=['drop','drop_dedup'], buckets_select=['90:101'], ngram='Unigram', column='combo', topk=33)
        new_outlist = output_most_similar(s_tdnd, dsetset=['drop','drop_dedup'], buckets_select=['90:101'], ngram='Unigram', column='combo', topk=33)

    """
    outfile = os.path.join(s.logdir, outname)
    if type(dsetset) != list:
        if dsetset.upper() == 'ALL':
            dsetset = list(s.sim_results_max.keys())
        else:
            dsetset = [dsetset]
    if type(buckets_select) != list:
        buckets_select = [buckets_select]


    outlist = []
    for dset in dsetset:     # for each eval dataset
        trainsets = s.sim_results_summary_thresh[dset][ngram][column].keys()
        for trainset in trainsets:
            buckets = s.sim_results_summary_thresh[dset][ngram][column][trainset].keys()
            for bucket in buckets:
                if buckets_select == ['ALL'] or bucket in buckets_select:
                    if s.sim_results_summary_thresh[dset][ngram][column][trainset][bucket]['num_similar'] > 0:
                        bucketstr = '"' + bucket + '"'
                        outlist.append(f'{dset},{trainset},{bucketstr}')
                        details = s.sim_results_summary_thresh[dset][ngram][column][trainset][bucket]['sim_details']
                        scores = s.sim_results_summary_thresh[dset][ngram][column][trainset][bucket]['sim_scores']
                        top_indices = np.argsort(scores)[topk*-1:]
                        for ind in top_indices:
                            outstr = "TEST: " + details[ind][0].replace(',', '') + "," + \
                                     "TRAIN: " + details[ind][1].replace(',', '')
                            outlist.append(outstr)
    with open(outfile, 'w') as f:
        f.write('\r\n'.join(outlist))
    return outlist


def output_most_similar_detail(s, dsetset='ALL',ngram='Unigram', column='combo', 
                         outname = 'tmp_most_similar_dump_detail.txt', topk=10000):
    """ Basic not very pretty dump of most similar eval vs train pairs
    Usage: 
        new_outlist = output_most_similar_detail(s_uqa_summary, dsetset=eval_metrics.unifiedqa_unseen_4, ngram='Unigram', column='combo', topk=10000, outname = 'tmp_uqa_most_similar_dump_detail.txt')
        new_outlist = output_most_similar_detail(s_tdnd_summary, dsetset=eval_metrics.unifiedqa_unseen_4, ngram='Unigram', column='combo', topk=10000, outname = 'tmp_tdnd_most_similar_dump_detail.txt')

    """
    outfile = os.path.join(s.logdir, outname)
    if type(dsetset) != list:
        if dsetset.upper() == 'ALL':
            dsetset = list(s.sim_results_max.keys())
        else:
            dsetset = [dsetset]
    firstresultset = list(s.eval_results[dsetset[0]].keys())[0]
    outlist = ['Eval Dataset,Combo Score,Question Score,Ans Score,Eval Sample,Most Similar Train Sample,Train Dataset,Similarity Bucket']
    for dset in dsetset:     # for each eval dataset
        print(f'Processing {dset} ...')
        simscores_np = np.array(s.sim_results_max[dset]['max_sim_over_train']['sim_scores'][ngram][column])
        rev_indices = np.argsort(simscores_np)[topk*-1:][::-1]
        for ind in rev_indices:
            currsimscore = s.sim_results_max[dset]['max_sim_over_train']['sim_scores'][ngram][column][ind]
            currpred = s.eval_results[dset][firstresultset]['predictions'][ind].replace(',', '').strip()
            outstr = dset + ','
            outstr += str(currsimscore) + ','
            if ind < len(s.sim_results_max[dset]['max_sim_over_train']['sim_scores'][ngram]['text']):
                outstr += str(s.sim_results_max[dset]['max_sim_over_train']['sim_scores'][ngram]['text'][ind]) + ','
            else:
                outstr += '-1.0,'
            if ind < len(s.sim_results_max[dset]['max_sim_over_train']['sim_scores'][ngram]['label']):
                outstr += str(s.sim_results_max[dset]['max_sim_over_train']['sim_scores'][ngram]['label'][ind]) + ','
            else:
                outstr += '-1.0,'
            details = s.sim_results_max[dset]['max_sim_over_train']['sim_details'][ngram][column][ind]
            outstr += "TEST: " + details[0].replace(',', '') + "  Prediction: " + currpred + "," + \
                     "TRAIN: " + details[1].replace(',', '') + ","
            outstr += str(s.sim_results_max[dset]['max_sim_over_train']['sim_train_ds'][ngram][column][ind]) + ','
            bucket_bottom = s.thresh_buckets[0]    
            for bucket_top in s.thresh_buckets[1:]:
                if currsimscore >= bucket_bottom and currsimscore < bucket_top:
                    bucketstr = str(bucket_bottom) + ':' + str(bucket_top)
                    break
                bucket_bottom = bucket_top
            outstr += bucketstr                    
            outlist.append(outstr)                        
    with open(outfile, 'w') as f:
        f.write('\r\n'.join(outlist))
    return outlist


def run_sim_detail_reports(logdir, sim_results_file, model_results_file, training_subsets_list, add_uqa=True):
    """ Run just the similarity detail dump report against different subsets of the training datasets.
        Note:   Model_results_file will supply the predictions in the output but these will only be valid 
                for the particular combination of training datasets the model was trained against...
    Usage: 
        logdir='/data/thar011/out/unifiedqa_averages/s2s3s4_v2/'
        sim_results_file='/data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd/eval_test_train_similarities_semb_thresh-100.1.json'  #reformatted eval questions for ssvise train datasets
        model_results_file='/data/thar011/out/unifiedqa_bart_large_s4_v1_qasc_dev_facts/eval_metrics.json'
        training_subsets_list = [ ['strategy_qa'],
                                  ['strategy_qa_facts_dev_in_train_selfsvised'],
                                  ['qasc_dev_facts_selfsvised'],
                                  ['qasc_facts_selfsvised'],
                                  ['cwwv', 'atomic'],
                                  ['strategy_qa', 'strategy_qa_facts_dev_in_train_selfsvised']
                                ]
        run_sim_detail_reports(logdir, sim_results_file, model_results_file, training_subsets_list)
        run_sim_detail_reports(logdir, sim_results_file, model_results_file, training_subsets_list, add_uqa=False)
    """
    if logdir[-1] != '/':
        logdir += '/'
    print(f'Reports will be out to {logdir}')
    os.makedirs(logdir, exist_ok=True)
    print(f'Loading similarity file {sim_results_file}...')
    sim_results = json.load(open(sim_results_file))
    results_list = [model_results_file]
    for training_subset in training_subsets_list:
        test_similarity_over = []
        out_list = ''
        if add_uqa:
            test_similarity_over = eval_metrics.unifiedqa_base_train.copy()
            out_list = 'unifiedqa_'
        out_list += '_'.join(training_subset)
        test_similarity_over.extend(training_subset.copy())
        print(f"Calculating eval dataset similarity to training subset: {test_similarity_over}")
        s_uqaplus_summary = SimilarityAggregator(sim_results, no_overlap_thresh=1000.0, results_list=results_list, 
                                                 compare_over=test_similarity_over, thresh_buckets = [0,60,90,101], logdir=logdir)
        new_outlist = output_most_similar_detail(s_uqaplus_summary, dsetset=eval_metrics.unifiedqa_unseen_4, ngram='Unigram', column='combo', 
                                                 topk=10000, outname = f'uqaplus_most_similar_dump_detail_us4_{out_list}.txt')
    return


def run_summary_thresh_reports(logdir, sim_results_file, results_list):
    """ Run the summary by sim threshold report for a set of model runs 
        The training mixture for each model run is read from the mixture key 
        in current-model-config.json file in each directory
        and this is the subset of training datasets similarity is calculated over.
    Usage:
        logdir='/data/thar011/out/unifiedqa_averages/s2s3s4s5_v1/'
        sim_results_file='/data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd/eval_test_train_similarities_semb_thresh-100.1.json'  #reformat
        results_list = ['/data/thar011/out/unifiedqa_bart_large_v3/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s2_sqa_sqafacts_v2_dev_in_train/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s2_sqa_sqafacts_v3_no_facts/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s2_sqa_sqafacts_v6_sqa_only/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s5_v2_sqafacts_dev_in_train_only/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s3_v1_cwwv/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s3_v2_cwwv_atomic/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s4_v2_cwwv_premask_atomic_premask/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s4_v3_cwwv_ssvise_atomic_ssvise/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s4_v1_qasc_dev_facts/eval_metrics.json',
                        '/data/thar011/out/unifiedqa_bart_large_s5_v1_qasc_facts/eval_metrics.json',
                       ]
        run_summary_thresh_reports(logdir, sim_results_file, results_list)
    """
    if logdir[-1] != '/':
        logdir += '/'
    print(f'Reports will be out to {logdir}')
    os.makedirs(logdir, exist_ok=True)
    print(f'Loading similarity file {sim_results_file}...')
    sim_results = json.load(open(sim_results_file))
    for result in results_list:
        result_name = result.split('/')[-2]
        print(f"Calculating for result file: {result_name} ...")
        result_dir = os.path.split(result)[0]
        with open(os.path.join(result_dir, 'current-model-config.json'), 'r') as ff:
            result_config = json.load(ff)
        mixture = result_config['mixture'] 
        test_similarity_over, mixture_key = eval_metrics.parse_mixture(mixture)
        test_similarity_over, mixture_key = eval_metrics.replace_sim(test_similarity_over, mixture_key)
        result_as_list = [result]
        s_uqaplus_summary = SimilarityAggregator(sim_results, no_overlap_thresh=1000.0, results_list=result_as_list,
                                                 compare_over=test_similarity_over, thresh_buckets = [0,60,90,101], logdir=logdir)
        s_uqaplus_summary.crosstab_x_train_y_evalbythresh(dsetset=eval_metrics.unifiedqa_unseen_4, output_metrics = ['mean_pred_score_str'], 
                                                          output_results = 'ALL', ngram='Unigram', column='combo', 
                                                          outname = f'crosstab_summary_us4_{result_name}{mixture_key}.txt')
    print("Finished!")
    return


def run_all_reports(logdir, sim_results_file, model_uqa_results_file, model_uqaplus_results_file):
    """ Runs all reports used in our paper and a few more...
    Usage: 
        logdir='/data/thar011/out/unifiedqa_averages/s2s3s4_v2/'
        sim_results_file='/data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd/eval_test_train_similarities_semb_thresh-100.1.json'  #reformat
        #sim_results_file='/data/thar011/out/unifiedqa_bart_large_v7indiv_digits_tdnd/eval_test_train_similarities_semb_thresh-100.1 (backup after ssvise but before use ssvise eval embeddings).json'
        run_all_reports(logdir=logdir,
                        sim_results_file=sim_results_file,
                        model_uqa_results_file='/data/thar011/out/unifiedqa_bart_large_v3/eval_metrics.json',
                        model_uqaplus_results_file='/data/thar011/out/unifiedqa_bart_large_s4_v1_qasc_dev_facts/eval_metrics.json')
    """
    if logdir[-1] != '/':
        logdir += '/'
    print(f'Reports will be out to {logdir}')
    os.makedirs(logdir, exist_ok=True)
    print(f'Loading similarity file {sim_results_file}...')
    sim_results = json.load(open(sim_results_file))
    results_list_uqa = [model_uqa_results_file]
    results_list_uqaplus = [model_uqaplus_results_file]

    # run these steps to produce summary by similarity bucket without breaking down by individual dataset for results generated prior to adding td and nd tasks to training regime
    s_uqa_summary = SimilarityAggregator(sim_results, no_overlap_thresh=1000.0, results_list=results_list_uqa, compare_over='UQA',
                                 thresh_buckets = [0,60,90,101], logdir=logdir)
    s_uqa_summary.crosstab_x_train_y_evalbythresh(dsetset=eval_metrics.unifiedqa_unseen_4, output_metrics = ['mean_pred_score_str'], 
                                                  output_results = 'ALL', ngram='Unigram', column='combo', outname = 'crosstab_uqa_summary_us4.txt')
    s_uqa_summary.crosstab_x_train_y_evalbythresh(dsetset=eval_metrics.unifiedqa_unseen_6, output_metrics = ['mean_pred_score_str'], 
                                                  output_results = 'ALL', ngram='Unigram', column='combo', outname = 'crosstab_uqa_summary_us6lowsimtdnd.txt')

    # run these steps to process summary results generated after adding td and nd tasks
    s_uqaplus_summary = SimilarityAggregator(sim_results, no_overlap_thresh=1000.0, results_list=results_list_uqaplus,
                                  thresh_buckets = [0,60,90,101], logdir=logdir)
    s_uqaplus_summary.crosstab_x_train_y_evalbythresh(dsetset=eval_metrics.unifiedqa_unseen_4, output_metrics = ['mean_pred_score_str'], 
                                                   output_results = 'ALL', ngram='Unigram', column='combo', outname = 'crosstab_uqaplus_summary_us4.txt')
    s_uqaplus_summary.crosstab_x_train_y_evalbythresh(dsetset=eval_metrics.unifiedqa_unseen_6, output_metrics = ['mean_pred_score_str'], 
                                                   output_results = 'ALL', ngram='Unigram', column='combo', outname = 'crosstab_uqaplus_summary_us6lowsimtdnd.txt')


    # Then run these steps to produce detail over UQA training sets, output csv files and prepare for the interleaving fn below:
    s_uqa = SimilarityAggregator(sim_results, no_overlap_thresh=60.0, results_list=results_list_uqa, compare_over='UQA',
                                 thresh_buckets = [0,60,90,101], logdir=logdir)
    s_uqa.crosstab_x_train_y_evalbythresh(dsetset=eval_metrics.unifiedqa_unseen_4, output_metrics = ['mean_pred_score_str'], 
                                          output_results = 'ALL', ngram='Unigram', column='combo', outname = 'crosstab_uqa_detail_us4.txt')

    # Then run these steps to produce detail over UQA+TDND training sets, output csv files with different metrics and prepare for the interleaving fn below:
    s_uqaplus = SimilarityAggregator(sim_results, no_overlap_thresh=60.0, results_list=results_list_uqaplus,
                                  thresh_buckets = [0,60,90,101], logdir=logdir)
    s_uqaplus.crosstab_x_train_y_evalbythresh(dsetset=eval_metrics.unifiedqa_unseen_4, output_metrics = ['mean_pred_score_str'], 
                                           output_results = 'ALL', ngram='Unigram', column='combo', outname = 'crosstab_uqaplus_detail_us4.txt')

    # Can run these steps to produce the combined output report (make sure you run the above for ['mean_pred_score_str'] last so .outlist contains the correct info..):
    new_outlist = output_interleave(s_uqa.outlist, s_uqaplus.outlist, choose_bucket = ['90:101'], outname=logdir+'interleave_90-101_us4.txt')
    new_outlist = output_interleave(s_uqa.outlist, s_uqaplus.outlist, choose_bucket = ['60:90'], outname=logdir+'interleave_60-90_us4.txt')
    new_outlist = output_interleave(s_uqa.outlist, s_uqaplus.outlist, choose_bucket = ['0:60'], outname=logdir+'interleave_0-60_us4.txt')


    # run these steps to produce the combined output report for us6:
    s_uqa.crosstab_x_train_y_evalbythresh(dsetset=eval_metrics.unifiedqa_unseen_6, output_metrics = ['mean_pred_score_str'], 
                                          output_results = 'ALL', ngram='Unigram', column='combo', outname = 'crosstab_uqa_detail_us6lowsimtdnd.txt')
    s_uqaplus.crosstab_x_train_y_evalbythresh(dsetset=eval_metrics.unifiedqa_unseen_6, output_metrics = ['mean_pred_score_str'], 
                                           output_results = 'ALL', ngram='Unigram', column='combo', outname = 'crosstab_uqaplus_detail_us6lowsimtdnd.txt')
    new_outlist = output_interleave(s_uqa.outlist, s_uqaplus.outlist, choose_bucket = ['90:101'], outname=logdir+'interleave_90-101_us6lowsimtdnd.txt')
    new_outlist = output_interleave(s_uqa.outlist, s_uqaplus.outlist, choose_bucket = ['60:90'], outname=logdir+'interleave_60-90_us6lowsimtdnd.txt')
    new_outlist = output_interleave(s_uqa.outlist, s_uqaplus.outlist, choose_bucket = ['0:60'], outname=logdir+'interleave_0-60_us6lowsimtdnd.txt')


    
    # Run this to produce the move vs stay analysis
    new_outlist = calc_pred_difference(s_uqa, s_uqaplus, dsetset=eval_metrics.unifiedqa_unseen_4, ngram='Unigram', column='combo', 
                                       outname = 'pred_diff_all_buckets_us4.txt')
    new_outlist = calc_pred_difference(s_uqa, s_uqaplus, dsetset=eval_metrics.unifiedqa_unseen_4, ngram='Unigram', column='combo', 
                                       outname = 'pred_diff_lowest_bucket_only_us4.txt', lowest_sim_bucket_only=True)            
    new_outlist = calc_pred_difference(s_uqa, s_uqaplus, dsetset=eval_metrics.unifiedqa_unseen_6, ngram='Unigram', column='combo', 
                                       outname = 'pred_diff_all_buckets_us6lowsimtdnd.txt')
    new_outlist = calc_pred_difference(s_uqa, s_uqaplus, dsetset=eval_metrics.unifiedqa_unseen_6, ngram='Unigram', column='combo', 
                                       outname = 'pred_diff_lowest_bucket_only_us6lowsimtdnd.txt', lowest_sim_bucket_only=True)            

    
    # Run this to produce sorted listing of most similar train sample to EACH test set sample..
    new_outlist = output_most_similar_detail(s_uqa_summary, dsetset=eval_metrics.unifiedqa_unseen_4, ngram='Unigram', column='combo', 
                                             topk=10000, outname = 'uqa_most_similar_dump_detail_us4.txt')
    new_outlist = output_most_similar_detail(s_uqaplus_summary, dsetset=eval_metrics.unifiedqa_unseen_4, ngram='Unigram', column='combo', 
                                             topk=10000, outname = 'uqaplus_most_similar_dump_detail_us4.txt')
    new_outlist = output_most_similar_detail(s_uqa_summary, dsetset=eval_metrics.unifiedqa_unseen_6, ngram='Unigram', column='combo', 
                                             topk=10000, outname = 'uqa_most_similar_dump_detail_us6lowsimtdnd.txt')
    new_outlist = output_most_similar_detail(s_uqaplus_summary, dsetset=eval_metrics.unifiedqa_unseen_6, ngram='Unigram', column='combo', 
                                             topk=10000, outname = 'uqaplus_most_similar_dump_detail_us6lowsimtdnd.txt')

    # Run this to produce dump of top 3 most similar in each bucket
    new_outlist = output_most_similar(s_uqa, dsetset=eval_metrics.unifiedqa_unseen_4, buckets_select='ALL', ngram='Unigram', column='combo', 
                                      topk=3, outname='uqa_mostsimilar_us4.txt')
    new_outlist = output_most_similar(s_uqaplus, dsetset=eval_metrics.unifiedqa_unseen_4, buckets_select='ALL', ngram='Unigram', column='combo', 
                                      topk=3, outname='uqaplus_mostsimilar_us4.txt')
    print('Finished!')
    return    


def check_duplicates(s, dsetset='ALL', ngram='Unigram', column='combo', 
                         outname = 'tmp_duplicate_counts.txt'):
    """ Check for duplicates in dev and test sets
    """
    outfile = os.path.join(s.logdir, outname)
    if type(dsetset) != list:
        if dsetset.upper() == 'ALL':
            dsetset = list(s.sim_results_max.keys())
        else:
            dsetset = [dsetset]
    
    outlist = ['Eval Dataset,Number of samples, Number of Unique Samples, Number of Duplicates']
    for dset in dsetset:     # for each eval dataset
        details = s.sim_results_max[dset]['max_sim_over_train']['sim_details'][ngram][column]
        details = [d[0] for d in details]
        unique = set(details)
        outstr = f'{dset},{len(details)},{len(unique)},{len(details)-len(unique)}'
        outlist.append(outstr)
    with open(outfile, 'w') as f:
        f.write('\r\n'.join(outlist))
    return outlist
        

class UQADataset:
    def __init__(self):
        pass

    def _load(self, datafile):
        result = []
        for row in datafile:
            q = eval_metrics.normalize_answer(row['question'])
            if type(row['answer']) != list:
                a = eval_metrics.normalize_answer(row['answer'])
            else:    
                a = eval_metrics.normalize_answer(row['answer'][0])
            result.append({"text": q, "label": a})
        return result


    def load(self, train_file, test_file):
        return pd.DataFrame(self._load(train_file)), pd.DataFrame(self._load(test_file))

    
    def _load_nonorm(self, datafile):
        result = []
        for row in datafile:
            q = row['question'].strip()
            if type(row['answer']) != list:
                a = row['answer'].strip()
            else:    
                a = row['answer'][0].strip()
            result.append({"text": q, "label": a})
        return result


    def load_nonorm(self, train_file, test_file):
        return pd.DataFrame(self._load_nonorm(train_file)), pd.DataFrame(self._load_nonorm(test_file))
    
    
    def is_ssvised(self, datafile):
        """ if answer is '' then consider entire file self supervised ie no labels """
        test_ans = datafile[0]['answer']
        if type(test_ans) == list:
            test_ans = test_ans[0]
        return test_ans.strip() == ''
        

    def run_similarity_comparer(self, trainfile, testfile, add_combo=True, answer_thresh=40.0, 
                                answer_col='label', question_thresh=0.0, use_cosine='bow',
                                train_emb=None, test_emb=None):
        ssvise = False
        if self.is_ssvised(trainfile) or self.is_ssvised(testfile):
            ssvise = True
        
        if use_cosine in ['bow', 'f1']:
            train_df, test_df = self.load(trainfile, testfile)  #
            result_score, result_detail = SimilarityEvaluator().run(test_df, train_df, column=["text", "label"], 
                                                                    add_combo=add_combo, answer_thresh=answer_thresh, 
                                                                    answer_col=answer_col, question_thresh=question_thresh, 
                                                                    use_cosine=use_cosine, ssvise=ssvise) 
        elif use_cosine == 'emb':
            train_df, test_df = self.load_nonorm(trainfile, testfile)  #
            train_emb_rename = {'text': train_emb['question'], 'label': train_emb['answer']}
            test_emb_rename = {'text': test_emb['question'], 'label': test_emb['answer']}

            result_score, result_detail = SimilarityEvaluator().run(test_df, train_df, column=["text", "label"], 
                                                                    add_combo=add_combo, answer_thresh=answer_thresh, 
                                                                    answer_col=answer_col, question_thresh=question_thresh, 
                                                                    use_cosine=use_cosine, ssvise=ssvise, 
                                                                    train_emb=train_emb_rename, test_emb=test_emb_rename )
        else:
            print(f"Unknown similarity comparison type: {use_cosine}")
        return result_score, result_detail





