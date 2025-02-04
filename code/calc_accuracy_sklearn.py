"""
Classify datasets using Naive Bayes and other algortihms
Author: Tim Hartill

"""
# Adapted from https://scikit-learn.org/0.24/auto_examples/text/plot_document_classification_20newsgroups.html
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Lars Buitinck
# License: BSD 3 clause

import logging
import numpy as np
import argparse
import os
import sys
from time import time
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics

from utils import load_jsonl, load_uqa_supervised, convert_pararules


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument("--print_report",
              action="store_true",
              help="Print a detailed classification report.")
parser.add_argument("--select_chi2",
              default=0, type=int,
              help="Select some number of features using a chi-squared test")
parser.add_argument("--print_cm",
              action="store_true", 
              help="Print the confusion matrix.")
parser.add_argument("--print_top10",
              action="store_true", 
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
parser.add_argument("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
parser.add_argument("--use_count",
              action="store_true",
              help="Use CountVectorizer.")
parser.add_argument("--n_features",
              type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")
parser.add_argument("--stopwords", default="NONE", type=str, 
                    help="Stopwords: 'NONE' or 'english'") 
parser.add_argument("--max_df", default=1.0, type=float, 
                    help="0.0 - 1.0: Ignore terms with doc frequency higher than this.") 
parser.add_argument("--input_dir", default="newsgroups", type=str, 
                    help="directory of data or 'newgroups'")
parser.add_argument("--all_categories",
              action="store_true", 
              help="Whether to use all categories or not if modelling newsgroups.")
parser.add_argument("--filtered",
              action="store_true",
              help="If modelling newsgroups Remove newsgroup information that is easily overfit: "
                   "headers, signatures, and quoting.")
parser.add_argument("--all_classifiers",
              action="store_true", 
              help="Run all classifiers. Default is to run a subset.")
parser.add_argument("--output_dir", default="/data/thar011/out/unifiedqa_averages/classifier_results", type=str, 
                    help="directory of data or 'newgroups'")
parser.add_argument("--show_chart",
              action="store_true",
              help="Show the results chart.")
args = parser.parse_args()
if args.stopwords.lower() == 'none':
    args.stopwords = None
else:
    args.stopwords = args.stopwords.lower()
#args.print_report=True
#args.print_cm=True
#args.max_df = 0.5
#args.use_count = True
#args.input_dir='/data/thar011/data/unifiedqa/strategy_qa'
#args.input_dir = '/data/thar011/data/jiant_combined_datasets/pararules_depth-3ext-NatLang'
#args.input_dir = '/data/thar011/data/jiant_combined_datasets/conceptrules_v2_full'
#args.input_dir = '/data/thar011/data/jiant_combined_datasets/conceptrules_v2_simplified'
# args.input_dir='/data/thar011/data/unifiedqa/qasc'  # Number of classes: Train:4363 Dev:674 All Dev classes in Train: False 493 classes in dev that don't exist in train!
os.makedirs(args.output_dir, exist_ok=True)
dataset = os.path.split(args.input_dir)[-1]
print(__doc__)
parser.print_help()
print()
print('OPTIONS USED:', args)


def format_classification(questions_train, answers_train, questions_dev, answers_dev):
    """ format dataset into classification friendly format
    Return dicts data_train, data_test with keys:
    'data' = list len num_samples of text (questions)
    'target' = list len num_samples of 0-based class ids
    'target_names' = list of class names in order of class ids
    """
    target_names_train = set(answers_train)
    target_names_dev = set(answers_dev)
    is_same = target_names_train.issuperset(target_names_dev)
    print(f"Number of classes: Train:{len(target_names_train)} Dev:{len(target_names_dev)} All Dev classes in Train: {is_same}")
    target_names = list(target_names_train)
    if not is_same:
        print(f"WARNING: There are {len(target_names_dev.difference(target_names_train))} classes in dev that don't exist in train! These will be mapped to an '<<unknown>>' class..")
        target_names.append('<<unknown>>')
    target_dict = {k:v for v, k in enumerate(target_names)}
    target_train = []
    for answer in answers_train:
        class_id = target_dict.get(answer)
        target_train.append(class_id)
    target_dev = []
    for answer in answers_dev:
        class_id = target_dict.get(answer, -1)
        if class_id == -1:
            class_id = target_dict['<<unknown>>']
        target_dev.append(class_id)
    data_train = {'data': questions_train, 'target': target_train, 'target_names': target_names}
    data_test = {'data': questions_dev, 'target': target_dev, 'target_names': target_names}   
    return data_train, data_test
    

def load_uqa_data(indir):
    """ Load unifiedqa-formatted data
    """
    infile = os.path.join(indir, 'train.tsv')
    questions_train, answers_train = load_uqa_supervised(infile)
    infile = os.path.join(indir, 'dev.tsv')
    questions_dev, answers_dev = load_uqa_supervised(infile)
    data_train, data_test = format_classification(questions_train, answers_train, questions_dev, answers_dev) 
    return data_train, data_test


def load_pararules_data(indir, use_test=False):
    """ Load ParaRules formatted data - ParaRules, ConceptRules (full), ConceptRules (simplified)
    """
    infile = os.path.join(indir, 'train.jsonl')
    jsonl_train = load_jsonl(infile)
    if use_test:
        infile = os.path.join(indir, 'test.jsonl')
    else:
        infile = os.path.join(indir, 'dev.jsonl')
    jsonl_dev = load_jsonl(infile)
    questions_train, answers_train = convert_pararules(jsonl_train)
    questions_dev, answers_dev = convert_pararules(jsonl_dev)
    data_train, data_test = format_classification(questions_train, answers_train, questions_dev, answers_dev) 
    return data_train, data_test


if args.input_dir == 'newsgroups':
    if args.all_categories:
        categories = None
    else:
        categories = ['alt.atheism','talk.religion.misc','comp.graphics','sci.space',]    
    if args.filtered:
        remove = ('headers', 'footers', 'quotes')
    else:
        remove = ()
    print("Loading 20 newsgroups dataset for categories:")
    print(categories if categories else "all")   
    data_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42, remove=remove)  
    data_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42, remove=remove)
elif '/unifiedqa/' in args.input_dir:  # process a uqa formatted dataset - assumes train in train.tsv and test in dev.tsv..
    data_train, data_test = load_uqa_data(args.input_dir)
elif '/pararules' in args.input_dir or '/conceptrules' in args.input_dir:
    data_train, data_test = load_pararules_data(args.input_dir)
else:
    assert True==False, f"ERROR: Invalid input directory: {args.input_dir}"    
print('data loaded')
print(data_train.keys())  # dict_keys(['data', 'filenames', 'target_names', 'target', 'DESCR'])

# order of labels in `target_names` can be different from `categories`
target_names = data_train['target_names']


def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6


data_train_size_mb = size_mb(data_train['data'])
data_test_size_mb = size_mb(data_test['data'])

print("%d documents - %0.3fMB (training set)" % (
    len(data_train['data']), data_train_size_mb))
print("%d documents - %0.3fMB (test set)" % (
    len(data_test['data']), data_test_size_mb))
print("%d categories" % len(target_names))
print()

# split a training set and a test set
y_train, y_test = data_train['target'], data_test['target']

print("Extracting features from the training data using a sparse vectorizer")
t0 = time()
if args.use_count:
    vectorizer = CountVectorizer(max_df=args.max_df, stop_words= args.stopwords)
    X_train = vectorizer.fit_transform(data_train['data'])    
elif args.use_hashing:
    vectorizer = HashingVectorizer(stop_words= args.stopwords, alternate_sign=False,
                                   n_features=args.n_features)
    X_train = vectorizer.transform(data_train['data'])
else:
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=args.max_df,
                                 stop_words= args.stopwords)
    X_train = vectorizer.fit_transform(data_train['data'])
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_train.shape)
print()

print("Extracting features from the test data using the same vectorizer")
t0 = time()
X_test = vectorizer.transform(data_test['data'])
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_test.shape)
print()

# mapping from integer feature name to original token string
if args.use_hashing:
    feature_names = None
else:
    feature_names = vectorizer.get_feature_names()

if args.select_chi2:
    print("Extracting %d best features by a chi-squared test" %
          args.select_chi2)
    t0 = time()
    ch2 = SelectKBest(chi2, k=args.select_chi2)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    if feature_names:
        # keep selected feature names
        feature_names = [feature_names[i] for i
                         in ch2.get_support(indices=True)]
    print("done in %fs" % (time() - t0))
    print()

if feature_names:
    feature_names = np.asarray(feature_names)


def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."


# Benchmark classifiers
# ------------------------------------
# train and test the datasets with different classification models
# and get performance results for each model.
def benchmark(clf, name=''):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if args.print_top10 and feature_names is not None:
            print("top 10 keywords per class:")
            for i, label in enumerate(target_names):
                if len(clf.coef_[i]) >= 10:
                    topk = -10
                else:
                    topk = len(clf.coef_[i]) * -1    
                top10 = np.argsort(clf.coef_[i])[topk:]
                print(trim("%s: %s" % (label, " ".join(feature_names[top10]))))
        print()

    if args.print_report:
        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=target_names))

    if args.print_cm:
        print(f"confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

    print()
    if name == '':
        name = str(clf).split('(')[0]
    return name, score, train_time, test_time


results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"), # This classifier first converts the target values into {-1, 1} and then treats the problem as a regression task (multi-output regression in the multiclass case).
        (Perceptron(max_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(max_iter=50),
         "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(), "Random forest")):
    if args.all_classifiers or name in ["Ridge Classifier"]:
        print('=' * 80)
        print(name)
        results.append(benchmark(clf, name))

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear version of linear SVM model
    results.append(benchmark(LinearSVC(penalty=penalty, dual=False,
                                       tol=1e-3), 'LinearSVC '+penalty))
    if args.all_classifiers:
        # Train SGD model: linear support vector machine (SVM)
        results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=50,
                                               penalty=penalty), 'SGDClassifier '+penalty))

# Train SGD with Elastic Net penalty:  squared euclidean norm L2 or the absolute norm L1: a combination of both (Elastic Net)
# The model it fits can be controlled with the loss parameter; by default, it fits a linear support vector machine (SVM)
#if args.all_classifiers:
print('=' * 80)
print("Linear SVM with Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=50,
                                       penalty="elasticnet"), 'SGDClassifier LinearSVM Elastic-Net'))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid()))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01)))
if args.all_classifiers:
    results.append(benchmark(BernoulliNB(alpha=.01)))
    results.append(benchmark(ComplementNB(alpha=.1)))

    print('=' * 80)
    print("LinearSVC with L1-based feature selection")
    # The smaller C, the stronger the regularization.
    # The more regularization, the more sparsity.
    results.append(benchmark(Pipeline([
      ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                      tol=1e-3))),
      ('classification', LinearSVC(penalty="l2"))]), 'LinearSVC l1-based feat. selection'))


# %%
# Add plots
# ------------------------------------
# The bar plot indicates the accuracy, training time (normalized) and test time
# (normalized) of each classifier.
indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
outfile = 'classifier_summary_' + dataset + '.txt'
outfile = os.path.join(args.output_dir, outfile)
outlist = ['Classifier,Accuracy,Train (secs),Test (secs)']
for i, name in enumerate(clf_names):
    outstr = f"{name},{score[i]},{training_time[i]},{test_time[i]}"
    outlist.append(outstr)
with open(outfile, 'w') as f:    
    f.write('\r\n'.join(outlist))
print(f"Summary written to {outfile}")

if args.show_chart:
    training_time = np.array(training_time) / np.max(training_time)
    test_time = np.array(test_time) / np.max(test_time)
    
    plt.figure(figsize=(12, 8))
    plt.title(dataset + " Score")
    plt.barh(indices, score, .2, label="score", color='navy')
    plt.barh(indices + .3, training_time, .2, label="training time",
             color='c')
    plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)
    
    for i, c in zip(indices, clf_names):
        plt.text(-.3, i, c)
    
    plt.show()
