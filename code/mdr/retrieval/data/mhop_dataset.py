# Portions Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.

"""
Processed BeerQA sample:

sample = {"question": "Which genus contains more species, Ortegocactus or Eschscholzia?", 
 "answers": ["Eschscholzia"], 
 "id": "b3d50a40b29d4283609de1d3f426aebce198a0b2", 
 "type": "comparison", 
 "src": "hotpotqa", 
 "para_agg_map": {"Eschscholzia": 0, "Ortegocactus": 0}, 
 "bridge": ["Eschscholzia", "Ortegocactus"], 
 "pos_paras": [{"title": "Eschscholzia", "text": "Eschscholzia is a genus of 12 annual or perennial plants in the Papaveraceae (poppy) family. The genus was named after the Baltic German/Imperial Russian botanist Johann Friedrich von Eschscholtz (1793-1831). All species are native to Mexico or the southern United States. Leaves are deeply cut, glabrous and glaucous, mostly basal, though a few grow on the stem."}, {"title": "Ortegocactus", "text": "Ortegocactus macdougallii is a species of cactus and the sole species of the genus Ortegocactus. The plant has a greenish-gray epidermis and black spines. It is only known from Oaxaca, Mexico."}], 
 "neg_paras": [{"title": "Johann Friedrich von Eschscholtz", "text": "Johann Friedrich Gustav von Eschscholtz (1 November 1793 \u2013 7 May 1831) was a Baltic German physician, naturalist, and entomologist. He was one of the earliest scientific explorers of the Pacific region, making significant collections of flora and fauna in Alaska, California, and Hawaii.", "src": "hl"}, {"title": "Perennial plant", "text": "A perennial plant or simply perennial is a plant that lives more than two years. The term (\"per-\" + \"-ennial\", \"through the years\") is often used to differentiate a plant from shorter-lived annuals and biennials. The term is also widely used to distinguish plants with little or no woody growth from trees and shrubs, which are also technically perennials.", "src": "hl"}, {"title": "Flowering plant", "text": "The flowering plants, also known as Angiospermae, or Magnoliophyta, are the most diverse group of land plants, with 64 orders, 416 families, approximately 13,000 known genera and 300,000 known species. Like gymnosperms, angiosperms are seed-producing plants. They are distinguished from gymnosperms by characteristics including flowers, endosperm within the seeds, and the production of fruits that contain the seeds. Etymologically, \"angiosperm\" means a plant that produces seeds within an enclosure; in other words, a fruiting plant. The term comes from the Greek words (\"case\" or \"casing\") and (\"seed\").", "src": "hl"}, {"title": "Genus", "text": "A genus (plural genera) is a taxonomic rank used in the biological classification of living and fossil organisms, as well as viruses, in biology. In the hierarchy of biological classification, genus comes above species and below family. In binomial nomenclature, the genus name forms the first part of the binomial species name for each species within the genus.", "src": "hl"}, {"title": "Botany", "text": "Botany, also called plant science(s), plant biology or phytology, is the science of plant life and a branch of biology. A botanist, plant scientist or phytologist is a scientist who specialises in this field. The term \"botany\" comes from the Ancient Greek word (\"botan\u0113\") meaning \"pasture\", \"grass\", or \"fodder\"; is in turn derived from ( ), \"to feed\" or \"to graze\". Traditionally, botany has also included the study of fungi and algae by mycologists and phycologists respectively, with the study of these three groups of organisms remaining within the sphere of interest of the International Botanical Congress. Nowadays, botanists (in the strict sense) study approximately 410,000 species of land plants of which some 391,000 species are vascular plants (including approximately 369,000 species of flowering plants), and approximately 20,000 are bryophytes.", "src": "hl"}, {"title": "Annual plant", "text": "An annual plant is a plant that completes its life cycle, from germination to the production of seeds, within one growing season, and then dies. The length of growing seasons and period in which they take place vary according to geographical location, and may not correspond to the four traditional seasonal divisions of the year. With respect to the traditional seasons annual plants are generally categorized into summer annuals and winter annuals. Summer annuals germinate during spring or early summer and mature by autumn of the same year. Winter annuals germinate during the autumn and mature during the spring or summer of the following calendar year.", "src": "hl"}, {"title": "Papaveraceae", "text": "The Papaveraceae are an economically important family of about 42 genera and approximately 775 known species of flowering plants in the order Ranunculales, informally known as the poppy family. The family is cosmopolitan, occurring in temperate and subtropical climates (mostly in the northern hemisphere), but almost unknown in the tropics. Most are herbaceous plants, but a few are shrubs and small trees. The family currently includes two groups that have been considered to be separate families: Fumariaceae and Pteridophyllaceae.", "src": "hl"}, {"title": "Leaf", "text": "A leaf (plural leaves) is the principal lateral appendage of the vascular plant stem, usually borne above ground and specialized for photosynthesis. The leaves and stem together form the shoot. Leaves are collectively referred to as foliage, as in \"autumn foliage\". In most leaves, the primary photosynthetic tissue, the palisade mesophyll, is located on the upper side of the blade or lamina of the leaf but in some species, including the mature foliage of \"Eucalyptus\", palisade mesophyll is present on both sides and the leaves are said to be isobilateral. Most leaves are flattened and have distinct upper (') and lower (') surfaces that differ in color, hairiness, the number of stomata (pores that intake and output gases), the amount and structure of epicuticular wax and other features. Leaves are mostly green in color due to the presence of a compound called chlorophyll that is essential for photosynthesis as it absorbs light energy from the sun. A leaf with white patches or edges is called a variegated leaf.", "src": "hl"}, {"title": "Mammillaria", "text": "Intense studies of DNA of the genus are being conducted, with preliminary results published for over a hundred taxa, and this promising approach might soon end the arguments. Based on DNA research results, the genus does not seem to be monophyletic and is likely to be split into two large genera, one of them possibly including certain species of other closely related genera like \"Coryphantha, Ortegocactus\" and \"Neolloydia\".", "src": "qp"}, {"title": "Genus", "text": " When the generic name is already known from context, it may be shortened to its initial letter, for example \"C. lupus\" in place of \"Canis lupus\". Where species are further subdivided, the generic name (or its abbreviated form) still forms the leading portion of the scientific name, for example, \" \" for the domestic dog (when considered a subspecies of the gray wolf) in zoology, or as a botanical example, \" \" ssp. \" \" . Also, as visible in the above examples, the Latinised portions of the scientific names of genera and their included species (and infraspecies, where applicable) are, by convention, written in italics.", "src": "qp"}, {"title": "Genus", "text": "Moreover, genera should be composed of phylogenetic units of the same kind as other (analogous) genera.", "src": "qp"}, {"title": "Genus", "text": "In zoological usage, taxonomic names, including those of genera, are classified as \"available\" or \"unavailable\". Available names are those published in accordance with the International Code of Zoological Nomenclature and not otherwise suppressed by subsequent decisions of the International Commission on Zoological Nomenclature (ICZN); the earliest such name for any taxon (for example, a genus) should then be selected as the \"valid\" (i.e., current or accepted) name for the taxon in question.", "src": "qp"}, {"title": "Genus", "text": "The number of species in genera varies considerably among taxonomic groups. For instance, among (non-avian) reptiles, which have about 1180 genera, the most (>300) have only 1 species, ~360 have between 2 and 4 species, 260 have 5-10 species, ~200 have 11-50 species, and only 27 genera have more than 50 species. However, some insect genera such as the bee genera \"Lasioglossum\" and \"Andrena\" have over 1000 species each. The largest flowering plant genus, \"Astragalus\", contains over 3,000 species.", "src": "qp"}, {"title": "Species", "text": "In biology, a species ( ) is the basic unit of classification and a taxonomic rank of an organism, as well as a unit of biodiversity. A species is often defined as the largest group of organisms in which any two individuals of the appropriate sexes or mating types can produce fertile offspring, typically by sexual reproduction. Other ways of defining species include their karyotype, DNA sequence, morphology, behaviour or ecological niche. In addition, paleontologists use the concept of the chronospecies since fossil reproduction cannot be examined.", "src": "hl"}, {"title": "CACTUS", "text": "CACTUS (Converted Atmospheric Cherenkov Telescope Using Solar-2) was a ground-based, Air Cherenkov Telescope (ACT) located outside Daggett, California, near Barstow. It was originally a solar power plant called Solar Two, but was converted to an observatory starting in 2001. The first astronomical observations started in the fall of 2004. However, the facility had its last observing runs in November 2005 as funds for observational operations from the National Science Foundation were no longer available. The facility was operated by the University of California, Davis but owned by Southern California Edison.", "src": "hl"}, {"title": "Mexico", "text": "Mexico (Spanish: \"M\u00e9xico\" ] ( ) ; Nahuan languages: \"M\u0113xihco\"), officially the United Mexican States (Spanish: \"Estados Unidos Mexicanos\"; EUM ] ( ) ), is a country in the southern portion of North America. It is bordered to the north by the United States; to the south and west by the Pacific Ocean; to the southeast by Guatemala, Belize, and the Caribbean Sea; and to the east by the Gulf of Mexico. Mexico covers 1972550 km2 and has approximately 128,649,565 inhabitants, making it the world's 13th-largest country by area, 10th-most populous country, and most populous Spanish-speaking nation. It is a federation comprising 31 states and Mexico City, its capital city and largest metropolis. Other major urban areas include Guadalajara, Monterrey, Puebla, Toluca, Tijuana, Ciudad Ju\u00e1rez, and Le\u00f3n.", "src": "hl"}, {"title": "Oaxaca", "text": "Oaxaca ( , , ] ( ) ; from ] ( ) ), officially the \"Free and Sovereign State of Oaxaca\" ( ), is one of the 32 states which compose the Federative Entities of Mexico. It is divided into 570 municipalities, of which 418 (almost three quarters) are governed by the system of (customs and traditions) with recognized local forms of self-governance. Its capital city is Oaxaca de Ju\u00e1rez.", "src": "hl"}, {"title": "Epidermis (botany)", "text": "The epidermis (from the Greek \"\u1f10\u03c0\u03b9\u03b4\u03b5\u03c1\u03bc\u03af\u03c2\", meaning \"over-skin\") is a single layer of cells that covers the leaves, flowers, roots and stems of plants. It forms a boundary between the plant and the external environment. The epidermis serves several functions: it protects against water loss, regulates gas exchange, secretes metabolic compounds, and (especially in roots) absorbs water and mineral nutrients. The epidermis of most leaves shows dorsoventral anatomy: the upper (adaxial) and lower (abaxial) surfaces have somewhat different construction and may serve different functions. Woody stems and some other stem structures such as potato tubers produce a secondary covering called the periderm that replaces the epidermis as the protective covering.", "src": "hl"}]
 }    

Note AISO reader/reranker input format: [CLS] [YES] [NO] [NONE] q [SEP] t1 [SOP] p1 [SEP] t2 [SOP] p2 [SEP]
ADDITIONAL_SPECIAL_TOKENS = {
    "YES": "[unused0]",
    "NO": "[unused1]",
    "SOP": "[unused2]",
    "NONE": "[unused3]"
}
tokenizer_extratoks = AutoTokenizer.from_pretrained('roberta-base', use_fast=True,
                                              additional_special_tokens=list(ADDITIONAL_SPECIAL_TOKENS.values()))
tokenizer_extratoks.additional_special_tokens
Out[53]: ['[unused0]', '[unused1]', '[unused2]', '[unused3]']
tokenizer_extratoks.additional_special_tokens
Out[53]: ['[unused0]', '[unused1]', '[unused2]', '[unused3]']
tokenizer_extratoks.tokenize('The [unused0] [unused1] [unused2] [unused3] is what?') 
['The', 'Ġ', '[unused0]', 'Ġ', '[unused1]', 'Ġ', '[unused2]', 'Ġ', '[unused3]', 'Ġis', 'Ġwhat', '?']

Note MDR unified has linear off CLS token to predict stop logits the CE to predict stopY/N, not a separate [NONE] token placeholder
Note2 MDR qa model has BCE loss on linear off 1st token for rank score



"""
import torch
from torch.utils.data import Dataset
import json
import random
from html import unescape

#from .data_utils import collate_tokens

from utils import encode_text, encode_query_paras, get_para_idxs, flatten, collate_tokens

class MhopDataset_var(Dataset):
    """ Version of MhopDataset designed to work with mhop_loss_var
    output: {'q': [q, q_sp1, q_sp1_sp2, ..., q_sp1_.._spx], 'c': [sp1, sp2, .., spx], "neg": [neg1, neg2, ... negn], "act_hops": [sample#hops]} 
    """
    def __init__(self, args, tokenizer, data_path, train=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_q_len = args.max_q_len
        self.max_c_len = args.max_c_len
        self.max_q_sp_len = args.max_q_sp_len
        self.train = train
        self.data_path = data_path
        self.negs = args.num_negs
        self.max_hops = args.max_hops
        self.use_sentences = args.query_use_sentences
        self.prepend_query = args.query_add_titles
        self.random_multi_seq = args.random_multi_seq
        print(f"Using sentences in query instead of full para: {self.use_sentences}")
        if self.use_sentences:
            self.prepend_query = True
            print(f"Always prepending sentences in query with title irrespective of --query_add_titles: {self.prepend_query}")
        else:
            print(f"Prepending para text in query with title: {self.prepend_query}")
            
        print(f"Random para order for 'multi' type sequencing: {self.random_multi_seq}")
        print(f"Loading data from {data_path}")
        self.data = [json.loads(line) for line in open(data_path).readlines()]
        if train: 
            self.data = [s for s in self.data if len(s["neg_paras"]) >= 2]
        print(f"Total sample count {len(self.data)}")

    def encode_para(self, para, max_len): #TJH Added truncation=True to eliminate warning - alternates removal of a token from each seq in the pair to get down to max_len
        return encode_text(self.tokenizer, unescape(para["title"].strip()), text_pair=para["text"].strip(), max_input_length=max_len, truncation=True, padding=False, return_tensors="pt")
        #return self.tokenizer.encode_plus(para["title"].strip(), text_pair=para["text"].strip(), max_length=max_len, truncation=True, return_tensors="pt")
    
    def __getitem__(self, index):
        sample = self.data[index]
        if sample.get('src') is None: #if no src key assume this is MDR-formatted HPQA data and reformat
            if sample.get('bridge') is not None:
                sample['bridge'] = [ sample['bridge'] ]
            sample['src'] = 'hotpotqa'
            
        question = sample['question']
        if question.endswith("?"):
            question = question[:-1]
        #max_hops = 2 # max hops on any training/dev sample
        para_list = []
        if sample["type"] == "multi": 
            # eg sample['bridge'] = [['Ossian Elgström', 'Kristian Zahrtmann', 'Peder Severin Krøyer'], ['bananarama'], ['tango']]
            # means all paras from sample['bridge'][0] (but in any order) must come before sample['bridge'][1] which in turn (in any order if > 1 para) must come before sample['bridge'][2] ..
            para_idxs = get_para_idxs(sample["pos_paras"])  # {'titleA': idx in pos_paras,  'titleB': idx in pos_paras, ..}
            for step_paras_list in sample["bridge"]:
                if self.random_multi_seq and self.train:
                    random.shuffle(step_paras_list)  
                for p in step_paras_list:
                    para_list += [sample["pos_paras"][pidx] for pidx in para_idxs[p]]  # > 1 pidx if para is repeated in pos_paras with difft sentence labels. eg could occur with FEVER 
        elif sample["type"] == "comparison":
            #num_hops = 2 # this sample actual number of hops
            random.shuffle(sample["pos_paras"])
            start_para, bridge_para = sample["pos_paras"]
            para_list = [start_para, bridge_para]
        elif sample["type"] == "bridge":
            #num_hops = 2 # this sample actual number of hops
            if len(sample["bridge"]) > 1:  #preprocessing couldn't identify unique final para
                random.shuffle(sample["bridge"])
            for para in sample["pos_paras"]:
                if para["title"] != sample["bridge"][0]:
                    start_para = para
                else:
                    bridge_para = para
            para_list = [start_para, bridge_para]
        elif sample["type"].strip() == '': #single hop eg squad
            #num_hops = 1
            start_para = sample["pos_paras"][0]
            para_list = [start_para]
        else:
            assert False, f"ERROR in Dataset: file:{self.data_path} index: {index}. Invalid type: {sample['type']}"

        if self.train:
            random.shuffle(sample["neg_paras"])

        num_hops = len(para_list)
        num_negs = len(sample["neg_paras"])
        neg_list = []
        for i in range(self.negs):
            if i < num_negs:
                neg_list.append( self.encode_para(sample["neg_paras"][i], self.max_c_len) )
            else:
                neg_list.append( self.encode_para( {"title": "dummy", "text": "dummy " + str(i)} ) )
        
        num_to_fill = self.max_hops - len(para_list)
        assert num_to_fill >= 0, f"ERROR: Sample has {len(para_list)} positive paras but max_hops is only {self.max_hops}."
        for i in range(num_to_fill): # Fill para_list with negs in reverse neg order so usually different from those in neg_list
            if i < num_negs:
                para_list.append(sample["neg_paras"][(i+1)*-1])
            else:
                para_list.append( {"title": "dummy", "text": "dummy " + str(i*-1)} )
            
            
        #start_para_codes = self.encode_para(start_para, self.max_c_len)
        #bridge_para_codes = self.encode_para(bridge_para, self.max_c_len)
        para_list_codes = [self.encode_para(p, self.max_c_len) for p in para_list]

        #num_paras = len(para_list)
        q_codes = encode_text(self.tokenizer, question, text_pair=None, max_input_length=self.max_q_len, truncation=True, padding=False, return_tensors="pt")
        q_list_codes = [q_codes]
        query_paras = ''
        for i in range(self.max_hops-1):  #if 3 paras: encode: q+sp1, q+sp1+sp2 but not q+sp1+sp2+sp3. Note: queries with neg paras are ignored in loss calc..
            if not self.use_sentences or i > num_hops-1:  # No sentence annots for neg paras
 #               if para_list[i]['text'][-1] not in ['.', '?', '!']:  # Force full stop at end since nq and tqa paras are chunked and might end mid sentence.
 #                   para_list[i]['text'] += '.'
 #               query_paras += ' ' + para_list[i]['text']  
                query_paras += ' ' + encode_query_paras(para_list[i]['text'], para_list[i]['title'], 
                                                        use_sentences=False, prepend_title=self.prepend_query, title_sep=':') 
            else:  # encode sentence or title:sentence rather than full para text
                query_paras += ' ' + encode_query_paras(para_list[i]['text'], para_list[i]['title'],
                                                        para_list[i]['sentence_spans'], para_list[i]['sentence_labels'],
                                                        self.use_sentences, self.prepend_query, title_sep=':')
            q_list_codes.append( encode_text(self.tokenizer, question, text_pair=query_paras.strip(), max_input_length=self.max_q_sp_len, truncation=True, padding=False, return_tensors="pt") )
        
        #q_sp_codes = encode_text(self.tokenizer, question, text_pair=start_para["text"].strip(), max_input_length=self.max_q_sp_len, truncation=True, padding=False, return_tensors="pt")
        #q_sp_codes = self.tokenizer.encode_plus(question, text_pair=start_para["text"].strip(), max_length=self.max_q_sp_len, truncation=True, return_tensors="pt")  #TJH: for q = 'a', text_pair='b': {'input_ids': [0, 102, 2, 2, 428, 2], 'attention_mask': [1, 1, 1, 1, 1, 1]}
        #q_codes = self.tokenizer.encode_plus(question, max_length=self.max_q_len, truncation=True, return_tensors="pt") #TJH: for q = 'a': {'input_ids': [0, 102, 2], 'attention_mask': [1, 1, 1]}

        return {
                "q": q_list_codes,                          # [q, q_sp1, q_sp1_sp2, ..., q_sp1_.._spx-1]. for hpqa: [q_codes, q_sp1_codes]
                "c": para_list_codes,                       # [sp1, sp2, .., spx] for hpqa: [start_para_codes, bridge_para_codes]
                "neg": neg_list,                            # [neg1, neg2, ... negn]
                "act_hops": torch.LongTensor([num_hops])    # sample #hops
                }

    def __len__(self):
        return len(self.data)


def mhop_collate_var(samples, pad_id=0):
    """ Collate variable step version: 
        e.g. samples[i]['q']: [ [tensor[1,2,3], tensor[1,2], tensor[1,2]], .. ] -> [ tensor[[1,2,3], [1,2,0], [1,2,0]], .. ]

    NOTE:pad_id was not used in original mhop_collate - should be but _mask should still use 0...
    """
    if len(samples) == 0:
        return {}
    max_hops = len(samples[0]['q'])
    num_negs = len(samples[0]['neg'])
    batch = {
            'q_input_ids': [collate_tokens([s["q"][i]["input_ids"] for s in samples], pad_id) for i in range(max_hops)],
            'q_mask': [collate_tokens([s["q"][i]["attention_mask"] for s in samples], 0) for i in range(max_hops)],

            'c_input_ids': [collate_tokens([s["c"][i]["input_ids"] for s in samples], pad_id) for i in range(max_hops)],
            'c_mask': [collate_tokens([s["c"][i]["attention_mask"] for s in samples], 0) for i in range(max_hops)],

            'neg_input_ids': [collate_tokens([s["neg"][i]["input_ids"] for s in samples], pad_id) for i in range(len(samples[0]['neg']))],
            'neg_mask': [collate_tokens([s["neg"][i]["attention_mask"] for s in samples], 0) for i in range(num_negs)],

            'act_hops': collate_tokens([s["act_hops"] for s in samples], 0), #[s["act_hops"] for s in samples],
        }

    if "token_type_ids" in samples[0]["q"][0]:
        batch.update({
            'q_type_ids': [collate_tokens([s["q"][i]["token_type_ids"] for s in samples], 0) for i in range(max_hops)],
            'c_type_ids': [collate_tokens([s["c"][i]["token_type_ids"] for s in samples], 0) for i in range(max_hops)],
            'neg_type_ids': [collate_tokens([s["neg"][i]["token_type_ids"] for s in samples], 0) for i in range(num_negs)],
        })

    return batch




class MhopDataset(Dataset):

    def __init__(self,
        tokenizer,
        data_path,
        max_q_len,
        max_q_sp_len,
        max_c_len,
        train=False,
        ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_q_len = max_q_len
        self.max_c_len = max_c_len
        self.max_q_sp_len = max_q_sp_len
        self.train = train
        print(f"Loading data from {data_path}")
        self.data = [json.loads(line) for line in open(data_path).readlines()]
        if train: 

            #import pdb; pdb.set_trace()  #TJH removed

            # debug TODO: remove for final release  #TJH removed as per MDR repo issue https://github.com/facebookresearch/multihop_dense_retrieval/issues/13
            #for idx in range(len(self.data)):
            #    self.data[idx]["neg_paras"] = self.data[idx]["tfidf_neg"]


            self.data = [_ for _ in self.data if len(_["neg_paras"]) >= 2]
        print(f"Total sample count {len(self.data)}")

    def encode_para(self, para, max_len):
        return self.tokenizer.encode_plus(para["title"].strip(), text_pair=para["text"].strip(), max_length=max_len, return_tensors="pt")
    
    def __getitem__(self, index):
        sample = self.data[index]
        question = sample['question']
        if question.endswith("?"):
            question = question[:-1]
        if sample["type"] == "comparison":
            random.shuffle(sample["pos_paras"])
            start_para, bridge_para = sample["pos_paras"]
        else:
            for para in sample["pos_paras"]:
                if para["title"] != sample["bridge"]:
                    start_para = para
                else:
                    bridge_para = para
        if self.train:
            random.shuffle(sample["neg_paras"])

        start_para_codes = self.encode_para(start_para, self.max_c_len)
        bridge_para_codes = self.encode_para(bridge_para, self.max_c_len)
        neg_codes_1 = self.encode_para(sample["neg_paras"][0], self.max_c_len)
        neg_codes_2 = self.encode_para(sample["neg_paras"][1], self.max_c_len)

        #TJH: tokenizer.encode_plus(['a','b'], max_length=8) = same as 'text_pair' form: {'input_ids': [0, 102, 2, 2, 428, 2], 'attention_mask': [1, 1, 1, 1, 1, 1]}
        #TJH: tokenizer.encode_plus(['a','b', 'c'], max_length=8): ERROR
        #TJH: tokenizer.encode_plus('</s></s>'.join(['a', 'b','c']), max_length=16) correct but watch len of each: {'input_ids': [0, 102, 2, 2, 428, 2, 2, 438, 2], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
        q_sp_codes = self.tokenizer.encode_plus(question, text_pair=start_para["text"].strip(), max_length=self.max_q_sp_len, return_tensors="pt")  #TJH: for q = 'a', text_pair='b': {'input_ids': [0, 102, 2, 2, 428, 2], 'attention_mask': [1, 1, 1, 1, 1, 1]}
        q_codes = self.tokenizer.encode_plus(question, max_length=self.max_q_len, return_tensors="pt") #TJH: for q = 'a': {'input_ids': [0, 102, 2], 'attention_mask': [1, 1, 1]}

        return {
                "q_codes": q_codes,                     #TJH q only
                "q_sp_codes": q_sp_codes,               #TJH q + start para text w/o title
                "start_para_codes": start_para_codes,   #TJH start para only incl title
                "bridge_para_codes": bridge_para_codes, #TJH 2nd para only incl title
                "neg_codes_1": neg_codes_1,             #TJH random neg para 1 incl title
                "neg_codes_2": neg_codes_2,             #TJH 2nd random neg para incl title
                }

    def __len__(self):
        return len(self.data)
    

def mhop_collate(samples, pad_id=0):
    if len(samples) == 0:
        return {}
    
    batch = {
            'q_input_ids': collate_tokens([s["q_codes"]["input_ids"].view(-1) for s in samples], 0),
            'q_mask':collate_tokens([s["q_codes"]["attention_mask"].view(-1) for s in samples], 0),

            'q_sp_input_ids': collate_tokens([s["q_sp_codes"]["input_ids"].view(-1) for s in samples], 0),
            'q_sp_mask':collate_tokens([s["q_sp_codes"]["attention_mask"].view(-1) for s in samples], 0),

            'c1_input_ids': collate_tokens([s["start_para_codes"]["input_ids"] for s in samples], 0),
            'c1_mask': collate_tokens([s["start_para_codes"]["attention_mask"] for s in samples], 0),
                
            'c2_input_ids': collate_tokens([s["bridge_para_codes"]["input_ids"] for s in samples], 0),
            'c2_mask': collate_tokens([s["bridge_para_codes"]["attention_mask"] for s in samples], 0),

            'neg1_input_ids': collate_tokens([s["neg_codes_1"]["input_ids"] for s in samples], 0),
            'neg1_mask': collate_tokens([s["neg_codes_1"]["attention_mask"] for s in samples], 0),
            
            'neg2_input_ids': collate_tokens([s["neg_codes_2"]["input_ids"] for s in samples], 0),
            'neg2_mask': collate_tokens([s["neg_codes_2"]["attention_mask"] for s in samples], 0),
            
        }

    if "token_type_ids" in samples[0]["q_codes"]:
        batch.update({
            'q_type_ids': collate_tokens([s["q_codes"]["token_type_ids"].view(-1) for s in samples], 0),
            'c1_type_ids': collate_tokens([s["start_para_codes"]["token_type_ids"] for s in samples], 0),
            'c2_type_ids': collate_tokens([s["bridge_para_codes"]["token_type_ids"] for s in samples], 0),
            "q_sp_type_ids": collate_tokens([s["q_sp_codes"]["token_type_ids"].view(-1) for s in samples], 0),
            'neg1_type_ids': collate_tokens([s["neg_codes_1"]["token_type_ids"] for s in samples], 0),
            'neg2_type_ids': collate_tokens([s["neg_codes_2"]["token_type_ids"] for s in samples], 0),
        })

    if "sent_ids" in samples[0]["start_para_codes"]:
        batch["c1_sent_target"] = collate_tokens([s["start_para_codes"]["sent_ids"] for s in samples], -1)
        batch["c1_sent_offsets"] = collate_tokens([s["start_para_codes"]["sent_offsets"] for s in samples], 0),

    return batch
