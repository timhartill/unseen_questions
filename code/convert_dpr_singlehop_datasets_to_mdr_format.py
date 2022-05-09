#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 17:11:34 2022

@author: tim hartill

Convert DPR versions of single hop datasets to "MDR" format to faciliate adding to training to bolster SQUAD-open retrieval performance

First download copies of the relevant files using the links found in https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py
Then extract to their json form.

"""

import os
import json
import random
import numpy as np
from html import unescape

import utils

from mdr_basic_tokenizer_and_utils import SimpleTokenizer, para_has_answer


OUTDIR_NQTQA = '/home/thar011/data/DPR/'
DPR_NQ_DEV = '/home/thar011/data/DPR/biencoder-nq-dev.json'
#DPR_NQ_TRAIN = '/home/thar011/data/DPR/biencoder-nq-train.json'
DPR_NQ_TRAIN_ADV_NEGS = '/home/thar011/data/DPR/nq-train-dense-results_as_input_with_gold.json'

DPR_TQA_DEV = '/home/thar011/data/DPR/triviaqa-dev_new.json'
DPR_TQA_TRAIN = '/home/thar011/data/DPR/triviaqa-train_new.json'

#DPR_TREC_DEV = '/home/thar011/data/DPR/curatedtrec-dev.json'
#DPR_TREC_TRAIN = '/home/thar011/data/DPR/curatedtrec-train.json'
#DPR_WQ_DEV = '/home/thar011/data/DPR/webquestions-dev.json'
#DPR_WQ_TRAIN = '/home/thar011/data/DPR/webquestions-train.json'

BQA_TRAIN = '/home/thar011/data/beerqa/beerqa_train_v1.0_with_neg_v0.jsonl'
BQA_DEV = '/home/thar011/data/beerqa/beerqa_dev_v1.0_with_neg_v0.jsonl'
BQA_QASVAL = '/home/thar011/data/beerqa/beerqa_qas_val.jsonl'

simple_tokenizer = SimpleTokenizer()


nq_dev = json.load(open(DPR_NQ_DEV)) #6515 [ dict_keys(['dataset', 'question', 'answers', 'positive_ctxs', 'negative_ctxs', 'hard_negative_ctxs']) ]
#nq_train = json.load(open(DPR_NQ_TRAIN)) #58880 [dict_keys(['dataset', 'question', 'answers', 'positive_ctxs', 'negative_ctxs', 'hard_negative_ctxs'])]
nq_train_adv = json.load(open(DPR_NQ_TRAIN_ADV_NEGS))  # 69639  dict_keys(['question', 'answers', 'negative_ctxs', 'hard_negative_ctxs', 'positive_ctxs'])

score_1000_test = [n for n in nq_dev if n['positive_ctxs'][0]['score'] != 1000] # all 1st entires have score 1000
score_1000_test = [n for n in nq_dev if len( n['positive_ctxs'])>1 and n['positive_ctxs'][1]['score'] == 1000] # None of the 2nd paras score 1000
score_1000_test = [n for n in nq_dev if n['answers'][0] not in n['positive_ctxs'][0]['text']]  #894 but appear to be differences in comma placement

#TODO Use score 1000 entry as the positive for NQ
#  nq_train_adv is newer train set, gives perf boost. But why difft # training samples?

dev_questions = set([n['question'] for n in nq_dev])
train_adv_questions = set([n['question'] for n in nq_train_adv])

overlap = train_adv_questions.intersection(dev_questions)  # set() so use nq_train_adv

multi_answer_test = [n for n in nq_dev if len(n['answers']) > 1]  # 696 have >1 answer
multi_answer_test = [n for n in nq_train_adv if len(n['answers']) > 1] # 7190 have >1 answer

len(nq_train_adv[0]['positive_ctxs']) #6
len(nq_train_adv[0]['negative_ctxs']) #0
len(nq_train_adv[0]['hard_negative_ctxs']) #30

any_neg_ctxs = [n for n in nq_train_adv if len(n['negative_ctxs']) > 0]  # None so use hard_negative_ctxs!

no_hard_negs = [n for n in nq_train_adv if len(n['hard_negative_ctxs']) == 0]  # 3  maybe just exclude these or fill with randoms
few_hard_negs = [n for n in nq_train_adv if len(n['hard_negative_ctxs']) < 10]  # 28 maybe just exclude these

no_hard_negs = [n for n in nq_dev if len(n['hard_negative_ctxs']) == 0]  # 7
few_hard_negs = [n for n in nq_dev if len(n['hard_negative_ctxs']) < 10]  # 30 - top up with 'negative ctxs' Note: 'negative_ctxs are quite randon, hard_negative_ctxs are much closer

no_hard_negs = [n for n in nq_dev if len(n['negative_ctxs']) == 0]  # 0
few_hard_negs = [n for n in nq_dev if len(n['negative_ctxs']) < 10]  # 0


no_pos_ctx = [s for s in nq_dev if len(s['positive_ctxs']) == 0] #0
no_pos_ctx = [s for s in nq_train_adv if len(s['positive_ctxs']) == 0] #0

# 1 only:
ans_not_in_para0 = [s for s in nq_dev if not para_has_answer(s['answers'], "yes no " + s['positive_ctxs'][0]['title'] + ' ' + s['positive_ctxs'][0]['text'], simple_tokenizer)]
# 9 only:
ans_not_in_para0 = [s for s in nq_train_adv if not para_has_answer(s['answers'], "yes no " + s['positive_ctxs'][0]['title'] + ' ' + s['positive_ctxs'][0]['text'], simple_tokenizer)]


### TREC - SKIP #####
#trec_dev = json.load(open(DPR_TREC_DEV))  # 116  dict_keys(['dataset', 'question', 'answers', 'positive_ctxs', 'negative_ctxs', 'hard_negative_ctxs'])
#trec_train = json.load(open(DPR_TREC_TRAIN))  # 1125  dict_keys(['dataset', 'question', 'answers', 'positive_ctxs', 'negative_ctxs', 'hard_negative_ctxs'])

# note strange trec answer format for 1st dev sample - ['Long Island|New\\s?York|Roosevelt Field'] and passage format: 'Charles Lindbergh Charles Augustus Lindbergh (February 4, 1902 – August 26, 1974) was an American aviator, military officer, author, inventor, explorer, and environmental activist. At age 25 in 1927, he went from obscurity as a [[U.S. Air Mail]] pilot to instantaneous world fame by winning the [[Orteig Prize]]: making a nonstop flight from [[Roosevelt Field (airport)|Roosevelt Field]], [[Long Island]], [[New York (state)|New York]], to [[Paris]], France. Lindbergh covered the -hour, flight alone in a single-engine purpose-built [[Ryan Airline Company|Ryan]] [[monoplane]], the "[[Spirit of St. Louis]]". This was not the [[Transatlantic flight of Alcock and Brown|first flight between North America and'
# several train egs have invalid highest scoring positive para
# SKIP TREC!!

# WEBQUESTIONS Skip
#wq_dev = json.load(open(DPR_WQ_DEV))  # 278  dict_keys(['question', 'answers', 'dataset', 'positive_ctxs', 'negative_ctxs', 'hard_negative_ctxs'])
#wq_train = json.load(open(DPR_WQ_TRAIN))  #2474 dict_keys(['question', 'answers', 'dataset', 'positive_ctxs', 'negative_ctxs', 'hard_negative_ctxs'])

# 1st one had invalid highest scoring para - skip WQ!


# TriviaQA #########

tqa_dev = json.load(open(DPR_TQA_DEV))  # 8837  dict_keys(['dataset', 'question', 'answers', 'positive_ctxs', 'negative_ctxs', 'hard_negative_ctxs'])
tqa_train = json.load(open(DPR_TQA_TRAIN))  #78785  dict_keys(['dataset', 'question', 'answers', 'positive_ctxs', 'negative_ctxs', 'hard_negative_ctxs'])

#tqa_dev[1] has 0 positive_ctxs.. so has tqa_train[0]  NEED TO SKIP THESE


any_neg_ctxs = [n for n in tqa_train if len(n['negative_ctxs']) > 0 and len(n['positive_ctxs']) > 0]  # 0

no_hard_negs = [n for n in tqa_train if len(n['hard_negative_ctxs']) == 0 and len(n['positive_ctxs']) > 0]  # 33  maybe just exclude these or fill with randoms
few_hard_negs = [n for n in tqa_train if len(n['hard_negative_ctxs']) < 10 and len(n['positive_ctxs']) > 0]  # 139 maybe just exclude these

no_hard_negs = [n for n in tqa_dev if len(n['hard_negative_ctxs']) == 0 and len(n['positive_ctxs']) > 0]  # 3
few_hard_negs = [n for n in tqa_dev if len(n['hard_negative_ctxs']) < 10 and len(n['positive_ctxs']) > 0]  # 15 - top up with 'negative ctxs' Note: 'negative_ctxs are quite randon, hard_negative_ctxs are much closer

no_hard_negs = [n for n in tqa_dev if len(n['negative_ctxs']) == 0 and len(n['positive_ctxs']) > 0]  # 6760
few_hard_negs = [n for n in tqa_dev if len(n['negative_ctxs']) < 10 and len(n['positive_ctxs']) > 0]  # 6760

no_pos_ctx = [s for s in tqa_dev if len(s['positive_ctxs']) == 0] #2207
no_pos_ctx = [s for s in tqa_train if len(s['positive_ctxs']) == 0] #18372

# 1 only:
ans_not_in_para0 = [s for s in tqa_dev if len(s['positive_ctxs']) > 0 and not para_has_answer(s['answers'], "yes no " + s['positive_ctxs'][0]['title'] + ' ' + s['positive_ctxs'][0]['text'], simple_tokenizer)]
# 4 only:
ans_not_in_para0 = [s for s in tqa_train if len(s['positive_ctxs']) > 0 and not para_has_answer(s['answers'], "yes no " + s['positive_ctxs'][0]['title'] + ' ' + s['positive_ctxs'][0]['text'], simple_tokenizer)]



def save_train_dev_file(split, outdir, src='nq', filetype='train'):
    """ save the train/dev file (with pos/neg paras) and optionally the "qas_val" file (without paras)
    traindev format is jsonl:
    {"question": "Which genus contains more species, Ortegocactus or Eschscholzia?", 
     "answers": ["Eschscholzia"], 
     "id": "b3d50a40b29d4283609de1d3f426aebce198a0b2", # set to 'src'+ str(idx)
     "type": "comparison",  # set to ''
     "src": "hotpotqa",     # set to "nq" or "tqa"
     "para_agg_map": {"Eschscholzia": 0, "Ortegocactus": 0}, # set to {}
     "bridge": ["Eschscholzia", "Ortegocactus"],  # set to ['title of pos para']
     "pos_paras": [{"title": "Eschscholzia", "text": "Eschscholzia is a genus of 12 annual or perennial plants in the Papaveraceae (poppy) family. The genus was named after the Baltic German/Imperial Russian botanist Johann Friedrich von Eschscholtz (1793-1831). All species are native to Mexico or the southern United States. Leaves are deeply cut, glabrous and glaucous, mostly basal, though a few grow on the stem."}, 
                   {"title": "Ortegocactus", "text": "Ortegocactus macdougallii is a species of cactus and the sole species of the genus Ortegocactus. The plant has a greenish-gray epidermis and black spines. It is only known from Oaxaca, Mexico."}], 
     "neg_paras": [{"title": "Johann Friedrich von Eschscholtz", "text": "Johann Friedrich Gustav von Eschscholtz (1 November 1793 \u2013 7 May 1831) was a Baltic German physician, naturalist, and entomologist. He was one of the earliest scientific explorers of the Pacific region, making significant collections of flora and fauna in Alaska, California, and Hawaii.", "src": "hl"}, {"title": "Perennial plant", "text": "A perennial plant or simply perennial is a plant that lives more than two years. The term (\"per-\" + \"-ennial\", \"through the years\") is often used to differentiate a plant from shorter-lived annuals and biennials. The term is also widely used to distinguish plants with little or no woody growth from trees and shrubs, which are also technically perennials.", "src": "hl"}, {"title": "Flowering plant", "text": "The flowering plants, also known as Angiospermae, or Magnoliophyta, are the most diverse group of land plants, with 64 orders, 416 families, approximately 13,000 known genera and 300,000 known species. Like gymnosperms, angiosperms are seed-producing plants. They are distinguished from gymnosperms by characteristics including flowers, endosperm within the seeds, and the production of fruits that contain the seeds. Etymologically, \"angiosperm\" means a plant that produces seeds within an enclosure; in other words, a fruiting plant. The term comes from the Greek words (\"case\" or \"casing\") and (\"seed\").", "src": "hl"}, {"title": "Genus", "text": "A genus (plural genera) is a taxonomic rank used in the biological classification of living and fossil organisms, as well as viruses, in biology. In the hierarchy of biological classification, genus comes above species and below family. In binomial nomenclature, the genus name forms the first part of the binomial species name for each species within the genus.", "src": "hl"}, {"title": "Botany", "text": "Botany, also called plant science(s), plant biology or phytology, is the science of plant life and a branch of biology. A botanist, plant scientist or phytologist is a scientist who specialises in this field. The term \"botany\" comes from the Ancient Greek word (\"botan\u0113\") meaning \"pasture\", \"grass\", or \"fodder\"; is in turn derived from ( ), \"to feed\" or \"to graze\". Traditionally, botany has also included the study of fungi and algae by mycologists and phycologists respectively, with the study of these three groups of organisms remaining within the sphere of interest of the International Botanical Congress. Nowadays, botanists (in the strict sense) study approximately 410,000 species of land plants of which some 391,000 species are vascular plants (including approximately 369,000 species of flowering plants), and approximately 20,000 are bryophytes.", "src": "hl"}, {"title": "Annual plant", "text": "An annual plant is a plant that completes its life cycle, from germination to the production of seeds, within one growing season, and then dies. The length of growing seasons and period in which they take place vary according to geographical location, and may not correspond to the four traditional seasonal divisions of the year. With respect to the traditional seasons annual plants are generally categorized into summer annuals and winter annuals. Summer annuals germinate during spring or early summer and mature by autumn of the same year. Winter annuals germinate during the autumn and mature during the spring or summer of the following calendar year.", "src": "hl"}, {"title": "Papaveraceae", "text": "The Papaveraceae are an economically important family of about 42 genera and approximately 775 known species of flowering plants in the order Ranunculales, informally known as the poppy family. The family is cosmopolitan, occurring in temperate and subtropical climates (mostly in the northern hemisphere), but almost unknown in the tropics. Most are herbaceous plants, but a few are shrubs and small trees. The family currently includes two groups that have been considered to be separate families: Fumariaceae and Pteridophyllaceae.", "src": "hl"}, {"title": "Leaf", "text": "A leaf (plural leaves) is the principal lateral appendage of the vascular plant stem, usually borne above ground and specialized for photosynthesis. The leaves and stem together form the shoot. Leaves are collectively referred to as foliage, as in \"autumn foliage\". In most leaves, the primary photosynthetic tissue, the palisade mesophyll, is located on the upper side of the blade or lamina of the leaf but in some species, including the mature foliage of \"Eucalyptus\", palisade mesophyll is present on both sides and the leaves are said to be isobilateral. Most leaves are flattened and have distinct upper (') and lower (') surfaces that differ in color, hairiness, the number of stomata (pores that intake and output gases), the amount and structure of epicuticular wax and other features. Leaves are mostly green in color due to the presence of a compound called chlorophyll that is essential for photosynthesis as it absorbs light energy from the sun. A leaf with white patches or edges is called a variegated leaf.", "src": "hl"}, {"title": "Mammillaria", "text": "Intense studies of DNA of the genus are being conducted, with preliminary results published for over a hundred taxa, and this promising approach might soon end the arguments. Based on DNA research results, the genus does not seem to be monophyletic and is likely to be split into two large genera, one of them possibly including certain species of other closely related genera like \"Coryphantha, Ortegocactus\" and \"Neolloydia\".", "src": "qp"}, {"title": "Genus", "text": " When the generic name is already known from context, it may be shortened to its initial letter, for example \"C. lupus\" in place of \"Canis lupus\". Where species are further subdivided, the generic name (or its abbreviated form) still forms the leading portion of the scientific name, for example, \" \" for the domestic dog (when considered a subspecies of the gray wolf) in zoology, or as a botanical example, \" \" ssp. \" \" . Also, as visible in the above examples, the Latinised portions of the scientific names of genera and their included species (and infraspecies, where applicable) are, by convention, written in italics.", "src": "qp"}, {"title": "Genus", "text": "Moreover, genera should be composed of phylogenetic units of the same kind as other (analogous) genera.", "src": "qp"}, {"title": "Genus", "text": "In zoological usage, taxonomic names, including those of genera, are classified as \"available\" or \"unavailable\". Available names are those published in accordance with the International Code of Zoological Nomenclature and not otherwise suppressed by subsequent decisions of the International Commission on Zoological Nomenclature (ICZN); the earliest such name for any taxon (for example, a genus) should then be selected as the \"valid\" (i.e., current or accepted) name for the taxon in question.", "src": "qp"}, {"title": "Genus", "text": "The number of species in genera varies considerably among taxonomic groups. For instance, among (non-avian) reptiles, which have about 1180 genera, the most (>300) have only 1 species, ~360 have between 2 and 4 species, 260 have 5-10 species, ~200 have 11-50 species, and only 27 genera have more than 50 species. However, some insect genera such as the bee genera \"Lasioglossum\" and \"Andrena\" have over 1000 species each. The largest flowering plant genus, \"Astragalus\", contains over 3,000 species.", "src": "qp"}, {"title": "Species", "text": "In biology, a species ( ) is the basic unit of classification and a taxonomic rank of an organism, as well as a unit of biodiversity. A species is often defined as the largest group of organisms in which any two individuals of the appropriate sexes or mating types can produce fertile offspring, typically by sexual reproduction. Other ways of defining species include their karyotype, DNA sequence, morphology, behaviour or ecological niche. In addition, paleontologists use the concept of the chronospecies since fossil reproduction cannot be examined.", "src": "hl"}, {"title": "CACTUS", "text": "CACTUS (Converted Atmospheric Cherenkov Telescope Using Solar-2) was a ground-based, Air Cherenkov Telescope (ACT) located outside Daggett, California, near Barstow. It was originally a solar power plant called Solar Two, but was converted to an observatory starting in 2001. The first astronomical observations started in the fall of 2004. However, the facility had its last observing runs in November 2005 as funds for observational operations from the National Science Foundation were no longer available. The facility was operated by the University of California, Davis but owned by Southern California Edison.", "src": "hl"}, {"title": "Mexico", "text": "Mexico (Spanish: \"M\u00e9xico\" ] ( ) ; Nahuan languages: \"M\u0113xihco\"), officially the United Mexican States (Spanish: \"Estados Unidos Mexicanos\"; EUM ] ( ) ), is a country in the southern portion of North America. It is bordered to the north by the United States; to the south and west by the Pacific Ocean; to the southeast by Guatemala, Belize, and the Caribbean Sea; and to the east by the Gulf of Mexico. Mexico covers 1972550 km2 and has approximately 128,649,565 inhabitants, making it the world's 13th-largest country by area, 10th-most populous country, and most populous Spanish-speaking nation. It is a federation comprising 31 states and Mexico City, its capital city and largest metropolis. Other major urban areas include Guadalajara, Monterrey, Puebla, Toluca, Tijuana, Ciudad Ju\u00e1rez, and Le\u00f3n.", "src": "hl"}, {"title": "Oaxaca", "text": "Oaxaca ( , , ] ( ) ; from ] ( ) ), officially the \"Free and Sovereign State of Oaxaca\" ( ), is one of the 32 states which compose the Federative Entities of Mexico. It is divided into 570 municipalities, of which 418 (almost three quarters) are governed by the system of (customs and traditions) with recognized local forms of self-governance. Its capital city is Oaxaca de Ju\u00e1rez.", "src": "hl"}, {"title": "Epidermis (botany)", "text": "The epidermis (from the Greek \"\u1f10\u03c0\u03b9\u03b4\u03b5\u03c1\u03bc\u03af\u03c2\", meaning \"over-skin\") is a single layer of cells that covers the leaves, flowers, roots and stems of plants. It forms a boundary between the plant and the external environment. The epidermis serves several functions: it protects against water loss, regulates gas exchange, secretes metabolic compounds, and (especially in roots) absorbs water and mineral nutrients. The epidermis of most leaves shows dorsoventral anatomy: the upper (adaxial) and lower (abaxial) surfaces have somewhat different construction and may serve different functions. Woody stems and some other stem structures such as potato tubers produce a secondary covering called the periderm that replaces the epidermis as the protective covering.", "src": "hl"}]}   
    
    
    qas_val format is jsonl:
    {"question": "Anti-infective drugs became more prominent after what war?", 
     "_id": "42a89ae12586b8d96120a37591cd39b3ac1c7ba3", # _id not id
     "answer": ["World War II"], # change from answers -> answer
     "sp": ["560876_8"], # set to []
     "type": "", 
     "src": "squad"  # set to "nq" or "tqa"
     }    
    
    """
    outfile_traindev = os.path.join(outdir, src + '_' + filetype + '_v1.0_with_neg_v0.jsonl')
    print(f"Train or dev file for this split only will be saved to: {outfile_traindev}")
    if filetype != 'train':
        outfile_qasval = os.path.join(outdir, src + '_qas_val.jsonl')
        print(f"qas_val file for this split only will be saved to: {outfile_qasval}")
    else:
        print("No qas_val file will be output.")

    out_traindev = []
    out_qasval = []
    for i, s in enumerate(split):
        if len(s['positive_ctxs']) == 0:
            continue
        idstr = src + str(i)
        pos_para = s['positive_ctxs'][0] 
        pos_para['title'] = unescape(pos_para['title'])  # likely unnecessary but just in case wasnt already escaped..
        bridge = [ pos_para['title'] ]
        pos_para = [ pos_para ]
        
        neg_paras = s['hard_negative_ctxs']
        if len(neg_paras) < 10:
            neg_paras += s['negative_ctxs']
        if len(neg_paras) < 10:
            num_to_find = 10 - len(neg_paras)
            for j in range(num_to_find):
                neg = random.choice(split)
                while neg['question'] == s['question'] or len(neg['positive_ctxs']) == 0:
                    neg = random.choice(split)
                neg_paras.append( neg['positive_ctxs'][0] )
        assert len(neg_paras) >= 10, f"Error idx:{i}  less than 10 neg paras"
        traindev_sample = {'question': s['question'], 'answers': s['answers'], 'id': idstr,
                           'type': '', 'src': src, 'para_agg_map': {}, 'bridge': bridge,
                           'pos_paras': pos_para, 'neg_paras': neg_paras}
        out_traindev.append(traindev_sample)
        if filetype != 'train':
            qasval_sample = {'question': s['question'], 'answer': s['answers'], '_id': idstr,
                             'type': '', 'src': src, 'sp': []}
            out_qasval.append(qasval_sample)
    print(f'traindev count: {len(out_traindev)}  qasval count: {len(out_qasval)}')
    utils.saveas_jsonl(out_traindev, outfile_traindev)
    if filetype != 'train':
         utils.saveas_jsonl(out_qasval, outfile_qasval)    
    return out_traindev, out_qasval
                

random.seed(42)        
out_dev_nq, out_qasval_nq = save_train_dev_file(nq_dev, OUTDIR_NQTQA, src='nq', filetype='dev')       
out_train_nq, _ = save_train_dev_file(nq_train_adv, OUTDIR_NQTQA, src='nq', filetype='train')   
    
out_dev_tqa, out_qasval_tqa = save_train_dev_file(tqa_dev, OUTDIR_NQTQA, src='tqa', filetype='dev')       
out_train_tqa, _ = save_train_dev_file(tqa_train, OUTDIR_NQTQA, src='tqa', filetype='train')       

beerqa_train = utils.load_jsonl(BQA_TRAIN)
beerqa_dev = utils.load_jsonl(BQA_DEV)
beerqa_qasval = utils.load_jsonl(BQA_QASVAL)

bqa_nq_train = beerqa_train + out_train_nq
utils.saveas_jsonl(bqa_nq_train, os.path.join(OUTDIR_NQTQA, 'bqa_nq_train_v1.0_with_neg_v0.jsonl'))

bqa_nq_tqa_train = beerqa_train + out_train_nq + out_train_tqa
utils.saveas_jsonl(bqa_nq_tqa_train, os.path.join(OUTDIR_NQTQA, 'bqa_nq_tqa_train_v1.0_with_neg_v0.jsonl'))

bqa_nq_dev = beerqa_dev + out_dev_nq
utils.saveas_jsonl(bqa_nq_dev, os.path.join(OUTDIR_NQTQA, 'bqa_nq_dev_v1.0_with_neg_v0.jsonl'))

bqa_nq_tqa_dev = beerqa_dev + out_dev_nq + out_dev_tqa
utils.saveas_jsonl(bqa_nq_tqa_dev, os.path.join(OUTDIR_NQTQA, 'bqa_nq_tqa_dev_v1.0_with_neg_v0.jsonl'))

bqa_nq_qasval = beerqa_qasval + out_qasval_nq
utils.saveas_jsonl(bqa_nq_qasval, os.path.join(OUTDIR_NQTQA, 'bqa_nq_qas_val.jsonl'))

bqa_nq__tqa_qasval = beerqa_qasval + out_qasval_nq + out_qasval_tqa
utils.saveas_jsonl(bqa_nq__tqa_qasval, os.path.join(OUTDIR_NQTQA, 'bqa_nq_tqa_qas_val.jsonl'))


##### create train/dev files without Squad in them ie hpqa nq tqa and hpqa nq

bqa_nq_dev = utils.load_jsonl( os.path.join(OUTDIR_NQTQA, 'bqa_nq_dev_v1.0_with_neg_v0.jsonl') )
bqa_nq_train = utils.load_jsonl( os.path.join(OUTDIR_NQTQA, 'bqa_nq_train_v1.0_with_neg_v0.jsonl') )

bqa_nq_tqa_dev = utils.load_jsonl( os.path.join(OUTDIR_NQTQA, 'bqa_nq_tqa_dev_v1.0_with_neg_v0.jsonl') )
bqa_nq_tqa_train = utils.load_jsonl( os.path.join(OUTDIR_NQTQA, 'bqa_nq_tqa_train_v1.0_with_neg_v0.jsonl') )


def strip_squad(split):
    """ Strip squad samples from a split
    """
    out_list = []
    for s in split:
        if s['src'] != 'squad':
            out_list.append( s )
    print(f"Original: {len(split)} Minus SQUAD:{len(out_list)}")
    return out_list


bqa_nosquad_nq_dev = strip_squad(bqa_nq_dev) # Original: 20636 Minus SQUAD:12504
utils.saveas_jsonl(bqa_nosquad_nq_dev, os.path.join(OUTDIR_NQTQA, 'bqa_nosquad_nq_dev_v1.0_with_neg_v0.jsonl'))

bqa_nosquad_nq_train = strip_squad(bqa_nq_train) # Original: 203682 Minus SQUAD:144397
utils.saveas_jsonl(bqa_nosquad_nq_train, os.path.join(OUTDIR_NQTQA, 'bqa_nosquad_nq_train_v1.0_with_neg_v0.jsonl'))


bqa_nosquad_nq_tqa_dev = strip_squad(bqa_nq_tqa_dev) # Original: 27396 Minus SQUAD:19264
utils.saveas_jsonl(bqa_nosquad_nq_tqa_dev, os.path.join(OUTDIR_NQTQA, 'bqa_nosquad_nq_tqa_dev_v1.0_with_neg_v0.jsonl'))

bqa_nosquad_nq_tqa_train = strip_squad(bqa_nq_tqa_train) # Original: 264095 Minus SQUAD:204810
utils.saveas_jsonl(bqa_nosquad_nq_tqa_train, os.path.join(OUTDIR_NQTQA, 'bqa_nosquad_nq_tqa_train_v1.0_with_neg_v0.jsonl'))


def calc_stats(split):
    """ Calculate stats, namely avg question length and number of unique support docs per source dataset
    """
    stats = {}
    for s in split:
        src = s['src']
        if stats.get(src) is None:
            stats[src] = {'q_len': [], 'unique_sps_title': set(), 'unique_sps_text':set(), 'unique_ans': set()}
        stats[src]['q_len'].append( len(s['question']) ) 
        stats[src]['unique_ans'].add( s['answers'][0] ) # only consider 1st answer if > 1
        for p in s['pos_paras']:
            stats[src]['unique_sps_title'].add( p['title'].lower() )
            stats[src]['unique_sps_text'].add( p['text'].lower() )
    for src in stats.keys():
        stats[src]['num_samples'] = len(stats[src]['q_len'])
        stats[src]['mean_q_len'] = float(np.mean(stats[src]['q_len']))
        stats[src]['num_unique_titles'] = len(stats[src]['unique_sps_title'])
        stats[src]['num_unique_paras'] = len(stats[src]['unique_sps_text'])
        stats[src]['num_unique_answers'] = len(stats[src]['unique_ans'])       
        print(f"{src}: #samples:{stats[src]['num_samples']}  AvgQlen:{stats[src]['mean_q_len']}  #Titles:{stats[src]['num_unique_titles']}  #Paras:{stats[src]['num_unique_paras']}  #Answers:{stats[src]['num_unique_answers']}")
    return stats


def calc_train_dev_overlap(stats_train, stats_dev):
    """ Calculate overlap of train/dev titles, paras and answers
    """
    overlaps = {}
    for src in stats_train.keys():
        if overlaps.get(src) is None:
            overlaps[src] = {'title_overlaps':0, 'para_overlaps':0, 'ans_overlaps':0}
        if stats_dev.get(src) is not None:
            overlaps[src]['title_overlaps'] = len(stats_train[src]['unique_sps_title'].intersection(stats_dev[src]['unique_sps_title']))
            overlaps[src]['title_overlaps_pct'] = overlaps[src]['title_overlaps'] / stats_train[src]['num_unique_titles']

            overlaps[src]['para_overlaps'] = len(stats_train[src]['unique_sps_text'].intersection(stats_dev[src]['unique_sps_text']))
            overlaps[src]['para_overlaps_pct'] = overlaps[src]['para_overlaps'] / stats_train[src]['num_unique_paras']

            overlaps[src]['ans_overlaps'] = len(stats_train[src]['unique_ans'].intersection(stats_dev[src]['unique_ans']))
            overlaps[src]['ans_overlaps_pct'] = overlaps[src]['ans_overlaps'] / stats_train[src]['num_unique_answers']
            print(f"{src}: {overlaps[src]}")
        else:
            print(f'{src} not present in dev split, skipping..')
    return overlaps


stats_bqa_nq_tqa_dev = calc_stats(bqa_nq_tqa_dev)
#hotpotqa: #samples:5989  AvgQlen:92.02704959091668  #Titles:11227  #Paras:11227  #Answers:5067
#squad: #samples:8132  AvgQlen:60.450319724545004  #Titles:54  #Paras:2179  #Answers:6826
#nq: #samples:6515  AvgQlen:47.07720644666155  #Titles:5456  #Paras:5984  #Answers:5349
#tqa: #samples:6760  AvgQlen:80.82204142011834  #Titles:6372  #Paras:6577  #Answers:5241

stats_bqa_nq_tqa_train = calc_stats(bqa_nq_tqa_train)
#squad: #samples:59285  AvgQlen:59.278266003204855  #Titles:385  #Paras:15686  #Answers:44323
#hotpotqa: #samples:74758  AvgQlen:105.09486610128683  #Titles:89463  #Paras:89478  #Answers:43825
#nq: #samples:69639  AvgQlen:47.32846537141544  #Titles:34879  #Paras:47431  #Answers:37538
#tqa: #samples:60413  AvgQlen:79.90389485706719  #Titles:42472  #Paras:50502  #Answers:25764

overlaps = calc_train_dev_overlap(stats_bqa_nq_tqa_train, stats_bqa_nq_tqa_dev)
#squad: {'title_overlaps': 0, 'para_overlaps': 0, 'ans_overlaps': 1253, 'title_overlaps_pct': 0.0, 'para_overlaps_pct': 0.0, 'ans_overlaps_pct': 0.028269747083906775}
#hotpotqa: {'title_overlaps': 6519, 'para_overlaps': 6519, 'ans_overlaps': 1863, 'title_overlaps_pct': 0.07286811307467891, 'para_overlaps_pct': 0.07285589753905988, 'ans_overlaps_pct': 0.04250998288648032}
#nq: {'title_overlaps': 3443, 'para_overlaps': 2575, 'ans_overlaps': 2622, 'title_overlaps_pct': 0.09871269245104504, 'para_overlaps_pct': 0.054289388796356815, 'ans_overlaps_pct': 0.06984921945761628}
#tqa: {'title_overlaps': 2825, 'para_overlaps': 1771, 'ans_overlaps': 3502, 'title_overlaps_pct': 0.06651440949331325, 'para_overlaps_pct': 0.035067918102253376, 'ans_overlaps_pct': 0.13592609843192052}



