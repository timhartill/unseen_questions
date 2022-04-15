#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 15:07:07 2022

@author: tim hartill

convert a beerqa dev file into the validation form expected by mdr_eval_mhop_retrieval_nativeamp.py, 
namely jsonl: 
    {"question": "Were Scott Derrickson and Ed Wood of the same nationality?", 
     "_id": "5a8b57f25542995d1e6f1371", 
     "answer": ["yes"], 
     "sp": ["Scott Derrickson", "Ed Wood"], 
     "type": "comparison"}
    
except in beerqa case 'sp' will be of form  [docid_paraidx, ...]

beerqa input file format is jsonl:
{"question": "Which genus contains more species, Ortegocactus or Eschscholzia?", 
 "answers": ["Eschscholzia"], 
 "id": "b3d50a40b29d4283609de1d3f426aebce198a0b2", 
 "type": "comparison",   # {'', 'bridge', 'comparison'}
 "src": "hotpotqa",      # {'hotpotqa', 'squad'}
 "para_agg_map": {"Eschscholzia": 0, "Ortegocactus": 0}, 
 "bridge": ["Eschscholzia", "Ortegocactus"], 
 "pos_paras": [{"title": "Eschscholzia", "text": "Eschscholzia is a genus of 12 annual or perennial plants in the Papaveraceae (poppy) family. The genus was named after the Baltic German/Imperial Russian botanist Johann Friedrich von Eschscholtz (1793-1831). All species are native to Mexico or the southern United States. Leaves are deeply cut, glabrous and glaucous, mostly basal, though a few grow on the stem."}, 
               {"title": "Ortegocactus", "text": "Ortegocactus macdougallii is a species of cactus and the sole species of the genus Ortegocactus. The plant has a greenish-gray epidermis and black spines. It is only known from Oaxaca, Mexico."}], 
 "neg_paras": [{"title": "Johann Friedrich von Eschscholtz", "text": "Johann Friedrich Gustav von Eschscholtz (1 November 1793 \u2013 7 May 1831) was a Baltic German physician, naturalist, and entomologist. He was one of the earliest scientific explorers of the Pacific region, making significant collections of flora and fauna in Alaska, California, and Hawaii.", "src": "hl"}, 
               {"title": "Perennial plant", "text": "A perennial plant or simply perennial is a plant that lives more than two years. The term (\"per-\" + \"-ennial\", \"through the years\") is often used to differentiate a plant from shorter-lived annuals and biennials. The term is also widely used to distinguish plants with little or no woody growth from trees and shrubs, which are also technically perennials.", "src": "hl"}, {"title": "Flowering plant", "text": "The flowering plants, also known as Angiospermae, or Magnoliophyta, are the most diverse group of land plants, with 64 orders, 416 families, approximately 13,000 known genera and 300,000 known species. Like gymnosperms, angiosperms are seed-producing plants. They are distinguished from gymnosperms by characteristics including flowers, endosperm within the seeds, and the production of fruits that contain the seeds. Etymologically, \"angiosperm\" means a plant that produces seeds within an enclosure; in other words, a fruiting plant. The term comes from the Greek words (\"case\" or \"casing\") and (\"seed\").", "src": "hl"}, {"title": "Genus", "text": "A genus (plural genera) is a taxonomic rank used in the biological classification of living and fossil organisms, as well as viruses, in biology. In the hierarchy of biological classification, genus comes above species and below family. In binomial nomenclature, the genus name forms the first part of the binomial species name for each species within the genus.", "src": "hl"}, {"title": "Botany", "text": "Botany, also called plant science(s), plant biology or phytology, is the science of plant life and a branch of biology. A botanist, plant scientist or phytologist is a scientist who specialises in this field. The term \"botany\" comes from the Ancient Greek word (\"botan\u0113\") meaning \"pasture\", \"grass\", or \"fodder\"; is in turn derived from ( ), \"to feed\" or \"to graze\". Traditionally, botany has also included the study of fungi and algae by mycologists and phycologists respectively, with the study of these three groups of organisms remaining within the sphere of interest of the International Botanical Congress. Nowadays, botanists (in the strict sense) study approximately 410,000 species of land plants of which some 391,000 species are vascular plants (including approximately 369,000 species of flowering plants), and approximately 20,000 are bryophytes.", "src": "hl"}, {"title": "Annual plant", "text": "An annual plant is a plant that completes its life cycle, from germination to the production of seeds, within one growing season, and then dies. The length of growing seasons and period in which they take place vary according to geographical location, and may not correspond to the four traditional seasonal divisions of the year. With respect to the traditional seasons annual plants are generally categorized into summer annuals and winter annuals. Summer annuals germinate during spring or early summer and mature by autumn of the same year. Winter annuals germinate during the autumn and mature during the spring or summer of the following calendar year.", "src": "hl"}, {"title": "Papaveraceae", "text": "The Papaveraceae are an economically important family of about 42 genera and approximately 775 known species of flowering plants in the order Ranunculales, informally known as the poppy family. The family is cosmopolitan, occurring in temperate and subtropical climates (mostly in the northern hemisphere), but almost unknown in the tropics. Most are herbaceous plants, but a few are shrubs and small trees. The family currently includes two groups that have been considered to be separate families: Fumariaceae and Pteridophyllaceae.", "src": "hl"}, {"title": "Leaf", "text": "A leaf (plural leaves) is the principal lateral appendage of the vascular plant stem, usually borne above ground and specialized for photosynthesis. The leaves and stem together form the shoot. Leaves are collectively referred to as foliage, as in \"autumn foliage\". In most leaves, the primary photosynthetic tissue, the palisade mesophyll, is located on the upper side of the blade or lamina of the leaf but in some species, including the mature foliage of \"Eucalyptus\", palisade mesophyll is present on both sides and the leaves are said to be isobilateral. Most leaves are flattened and have distinct upper (') and lower (') surfaces that differ in color, hairiness, the number of stomata (pores that intake and output gases), the amount and structure of epicuticular wax and other features. Leaves are mostly green in color due to the presence of a compound called chlorophyll that is essential for photosynthesis as it absorbs light energy from the sun. A leaf with white patches or edges is called a variegated leaf.", "src": "hl"}, {"title": "Mammillaria", "text": "Intense studies of DNA of the genus are being conducted, with preliminary results published for over a hundred taxa, and this promising approach might soon end the arguments. Based on DNA research results, the genus does not seem to be monophyletic and is likely to be split into two large genera, one of them possibly including certain species of other closely related genera like \"Coryphantha, Ortegocactus\" and \"Neolloydia\".", "src": "qp"}, {"title": "Genus", "text": " When the generic name is already known from context, it may be shortened to its initial letter, for example \"C. lupus\" in place of \"Canis lupus\". Where species are further subdivided, the generic name (or its abbreviated form) still forms the leading portion of the scientific name, for example, \" \" for the domestic dog (when considered a subspecies of the gray wolf) in zoology, or as a botanical example, \" \" ssp. \" \" . Also, as visible in the above examples, the Latinised portions of the scientific names of genera and their included species (and infraspecies, where applicable) are, by convention, written in italics.", "src": "qp"}, {"title": "Genus", "text": "Moreover, genera should be composed of phylogenetic units of the same kind as other (analogous) genera.", "src": "qp"}, {"title": "Genus", "text": "In zoological usage, taxonomic names, including those of genera, are classified as \"available\" or \"unavailable\". Available names are those published in accordance with the International Code of Zoological Nomenclature and not otherwise suppressed by subsequent decisions of the International Commission on Zoological Nomenclature (ICZN); the earliest such name for any taxon (for example, a genus) should then be selected as the \"valid\" (i.e., current or accepted) name for the taxon in question.", "src": "qp"}, {"title": "Genus", "text": "The number of species in genera varies considerably among taxonomic groups. For instance, among (non-avian) reptiles, which have about 1180 genera, the most (>300) have only 1 species, ~360 have between 2 and 4 species, 260 have 5-10 species, ~200 have 11-50 species, and only 27 genera have more than 50 species. However, some insect genera such as the bee genera \"Lasioglossum\" and \"Andrena\" have over 1000 species each. The largest flowering plant genus, \"Astragalus\", contains over 3,000 species.", "src": "qp"}, {"title": "Species", "text": "In biology, a species ( ) is the basic unit of classification and a taxonomic rank of an organism, as well as a unit of biodiversity. A species is often defined as the largest group of organisms in which any two individuals of the appropriate sexes or mating types can produce fertile offspring, typically by sexual reproduction. Other ways of defining species include their karyotype, DNA sequence, morphology, behaviour or ecological niche. In addition, paleontologists use the concept of the chronospecies since fossil reproduction cannot be examined.", "src": "hl"}, {"title": "CACTUS", "text": "CACTUS (Converted Atmospheric Cherenkov Telescope Using Solar-2) was a ground-based, Air Cherenkov Telescope (ACT) located outside Daggett, California, near Barstow. It was originally a solar power plant called Solar Two, but was converted to an observatory starting in 2001. The first astronomical observations started in the fall of 2004. However, the facility had its last observing runs in November 2005 as funds for observational operations from the National Science Foundation were no longer available. The facility was operated by the University of California, Davis but owned by Southern California Edison.", "src": "hl"}, {"title": "Mexico", "text": "Mexico (Spanish: \"M\u00e9xico\" ] ( ) ; Nahuan languages: \"M\u0113xihco\"), officially the United Mexican States (Spanish: \"Estados Unidos Mexicanos\"; EUM ] ( ) ), is a country in the southern portion of North America. It is bordered to the north by the United States; to the south and west by the Pacific Ocean; to the southeast by Guatemala, Belize, and the Caribbean Sea; and to the east by the Gulf of Mexico. Mexico covers 1972550 km2 and has approximately 128,649,565 inhabitants, making it the world's 13th-largest country by area, 10th-most populous country, and most populous Spanish-speaking nation. It is a federation comprising 31 states and Mexico City, its capital city and largest metropolis. Other major urban areas include Guadalajara, Monterrey, Puebla, Toluca, Tijuana, Ciudad Ju\u00e1rez, and Le\u00f3n.", "src": "hl"}, {"title": "Oaxaca", "text": "Oaxaca ( , , ] ( ) ; from ] ( ) ), officially the \"Free and Sovereign State of Oaxaca\" ( ), is one of the 32 states which compose the Federative Entities of Mexico. It is divided into 570 municipalities, of which 418 (almost three quarters) are governed by the system of (customs and traditions) with recognized local forms of self-governance. Its capital city is Oaxaca de Ju\u00e1rez.", "src": "hl"}, {"title": "Epidermis (botany)", "text": "The epidermis (from the Greek \"\u1f10\u03c0\u03b9\u03b4\u03b5\u03c1\u03bc\u03af\u03c2\", meaning \"over-skin\") is a single layer of cells that covers the leaves, flowers, roots and stems of plants. It forms a boundary between the plant and the external environment. The epidermis serves several functions: it protects against water loss, regulates gas exchange, secretes metabolic compounds, and (especially in roots) absorbs water and mineral nutrients. The epidermis of most leaves shows dorsoventral anatomy: the upper (adaxial) and lower (abaxial) surfaces have somewhat different construction and may serve different functions. Woody stems and some other stem structures such as potato tubers produce a secondary covering called the periderm that replaces the epidermis as the protective covering.", "src": "hl"}
               ]
 }

"""
import os
import json

from html import unescape

import utils

BEER_TITLE_SAVE = '/home/thar011/data/beerqa/enwiki-20200801-titledict-compgen.json'
#BEER_DEV = '/home/thar011/data/beerqa/beerqa_dev_v1.0.json'
BEER_DENSE_DEV = '/home/thar011/data/beerqa/beerqa_dev_v1.0_with_neg_v0.jsonl'
BEER_VAL_FILE_OUT = '/home/thar011/data/beerqa/beerqa_qas_val.jsonl'

print(f"Loading BeerQA file from {BEER_DENSE_DEV}...")
beerqa = utils.load_jsonl(BEER_DENSE_DEV)  #14121

print(f"Loading titledict from {BEER_TITLE_SAVE}...")
titledict = json.load(open(BEER_TITLE_SAVE))
print(f"title_dict loaded from {BEER_TITLE_SAVE} has {len(titledict)} titles...")

#titledict['chinnar wildlife sanctuary']  # [{'title': 'Chinnar Wildlife Sanctuary', 'id': '9642568', 'idx': 0}]


def map_title_case(hlink, titledict, id_type='id', verbose=False):
    """ Some titles in HPQA abstracts have incorrect casing. Attempt to map casing.
    hlink is a wiki doc title not necessarily from and hlink..
    titledict has key 'title' with entry(s) like: [{'title': 'Chinnar Wildlife Sanctuary', 'id': '9642568', 'idx': 0}]
    id_type = 'id' will return wiki doc id, id_type='idx' will return idx of this title in docs
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
            idx = tmap[0][id_type]
        else:
            for t in tmap:
                if hlink == t['title']: # exact match amongst candidates found, use that
                    return hlink, 'mok', t[id_type]
            hlink = tmap[0]['title']    # otherwise just return first
            idx = tmap[0][id_type]
            status = 'mc'
            if verbose:
                print(f"Hlink lower:{tlower} No exact match found so assigning first: {hlink}")
    return hlink, status, idx

#title_cased, status, doc_id = map_title_case('chinnar wildlife sanctuary', titledict, id_type='id', verbose=False)

def map_title_to_docid(beerqa, titledict):
    """ map "para_agg_map": {"Eschscholzia": 0, "Ortegocactus": 0} to new key "sp":[docid_paraidx]
    """
    for sample in beerqa:
        sp = []
        for title, para_idx in sample['para_agg_map'].items():
            title_cased, status, doc_id = map_title_case(title, titledict, id_type='id', verbose=False)
            sp.append( doc_id+'_'+str(para_idx) )
            if status in ['nf', 'mc']:
                print('Title: {title} has status {status}!')  # no such errors found
        sample['sp'] = sp
    return


def save_to_val_file(beerqa, outfile):
    print(f"saving to {outfile}..")
    out = []
    for sample in beerqa:
        out_sample = {'question': sample['question'], '_id': sample['id'], 'answer': sample['answers'], 
                      'sp': sample['sp'], 'type': sample['type'], 'src': sample['src']}
        out.append(out_sample)
    utils.saveas_jsonl(out, outfile)
    print('Finished save!')
    return


map_title_to_docid(beerqa, titledict)

save_to_val_file(beerqa, BEER_VAL_FILE_OUT)





