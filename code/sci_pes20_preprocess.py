#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 14:49:04 2024

@author: tim hartill

See https://github.com/allenai/peS2o

pes2o file preprocessing

outputs train/val files in jsonl format:
    
{'added': Value(dtype='string', id=None),
 'created': Value(dtype='string', id=None),
 'id': Value(dtype='string', id=None),
 'source': Value(dtype='string', id=None),
 'text': Value(dtype='string', id=None),
 'version': Value(dtype='string', id=None)}

"""

import os
import json

from datasets import load_dataset
from datasets import get_dataset_split_names
from datasets import load_dataset_builder

DS_HF = "allenai/peS2o"
OUTPUT_DIR = '/data/pes2o'

FILE_VAL = os.path.join(OUTPUT_DIR, 'peS2o_validation.jsonl')
FILE_TRAIN = os.path.join(OUTPUT_DIR, 'peS2o_train.jsonl')



def saveas_json(obj, file, mode="w", indent=5, add_nl=False):
    """ Save python object as json to file
    default mode w = overwrite file
            mode a = append to file
    indent = None: all json on one line
                0: pretty print with newlines between keys
                1+: pretty print with that indent level
    add_nl = True: Add a newline before outputting json. ie if mode=a typically indent=None and add_nl=True   
    Example For outputting .jsonl (note first line doesn't add a newline before):
        saveas_json(pararules_sample, DATA_DIR+'test_output.jsonl', mode='a', indent=None, add_nl=False)
        saveas_json(pararules_sample, DATA_DIR+'test_output.jsonl', mode='a', indent=None, add_nl=True)
          
    """
    with open(file, mode) as fp:
        if add_nl:
            fp.write('\n')
        json.dump(obj, fp, indent=indent)
    return True    


def saveas_jsonl(obj_list, file, initial_mode = 'w', verbose=True, update=5000):
    """ Save a list of json msgs as a .jsonl file of form:
        {json msg 1}
        {json msg 2}
        ...
        To overwrite exiting file use default initial_mode = 'w'. 
        To append to existing file set initial_mode = 'a'
    """
    if initial_mode == 'w':
        if verbose:
            print('Creating new file:', file)
        add_nl = False
    else:
        if verbose:
            print('Appending to file:', file)
        add_nl = True
    mode = initial_mode
    for i, json_obj in enumerate(obj_list):
            saveas_json(json_obj, file, mode=mode, indent=None, add_nl=add_nl)
            add_nl = True
            mode = 'a'
            if verbose:
                if i > 0 and i % update == 0:
                    print('Processed:', i)
    if verbose:
        print('Finished adding to:', file)        
    return True

def load_jsonl(file, verbose=True):
    """ Load a list of json msgs from a file formatted as 
           {json msg 1}
           {json msg 2}
           ...
    """
    if verbose:
        print('Loading json file: ', file)
    with open(file, "r") as f:
        all_json_list = f.read()
    all_json_list = all_json_list.split('\n')
    if verbose:
        print('JSON as text successfully loaded. Loading...')
    all_json_list = [json.loads(j) for j in all_json_list if j.strip() != '']
    if verbose:
        print(f'Text successfully converted to JSON. Number of json messages: {len(all_json_list)}')
    return all_json_list


#########################################

ds_builder = load_dataset_builder(DS_HF)
print(ds_builder.info.description)
print(ds_builder.info.dataset_size)  # 38972211
print(ds_builder.info.download_size) # 87129236480
print(ds_builder.info.features)  
"""{'added': Value(dtype='string', id=None),
 'created': Value(dtype='string', id=None),
 'id': Value(dtype='string', id=None),
 'source': Value(dtype='string', id=None),
 'text': Value(dtype='string', id=None),
 'version': Value(dtype='string', id=None)}
"""

get_dataset_split_names(DS_HF)  # ['train', 'validation']

ds = load_dataset(DS_HF, "v2")  # dict with all splits (typically datasets.Split.TRAIN and datasets.Split.TEST).
print(ds.cache_files)
print(ds.num_columns)
print(ds.num_rows)   # {'train': 38811179, 'validation': 161032}
print(ds.column_names)  # {'train': ['added', 'created', 'id', 'source', 'text', 'version'], 'validation': ['added', 'created', 'id', 'source', 'text', 'version']}
#type(ds['validation'][0])  # dict
#ds["train"][0:3]["text"]

v_list = [r for r in ds['validation']]
saveas_jsonl(v_list, FILE_VAL)


t_list = [r for r in ds['train']]  # note very slow
saveas_jsonl(t_list, FILE_TRAIN)   #  also very slow

print("finished download!")

##############################################
# local data exploration...
# tldr title precedes first \n\n. Paras within sections delimied by \n. For s2orc \n\n\n = end of abstract


v_list_local = load_jsonl("/media/tim/dl1storage/unseen_questions/pes2o/peS2o_validation.jsonl")
v_ptr = v_list_local[0]  # s2ag: title \n\n abstract para
"""{'added': '2022-12-11T16:08:35.184Z',
 'created': '2022-12-08T00:00:00.000Z',
 'id': '254531610',
 'source': 's2ag/valid',
 'text': 'Elimination of a right-sided accessory pathway using a videothoracoscopic approach after failed recurring catheter ablation: clinical case\n\nThe description of the clinical case presents a rare observation of a multi-stage approach to the treatment of right-sided accessory pathway. There are presented the results and features of successful epicardial ablation using a minimally invasive thoracoscopic approach, which made it possible to eliminate accessory pathways for right-sided epicardial localization after failed recurring cataract ablations.',
 'version': 'v2'}
"""
v_ptr = v_list_local[5] # s2ag: title \n\n Abstract para1\nAbs para 2\nAbs para 3
"""{'added': '2022-12-22T16:05:23.678Z',
 'created': '2022-12-20T00:00:00.000Z',
 'id': '254961670',
 'source': 's2ag/valid',
 'text': "The Lord's Supper Revisited\n\nThe Lord’s Supper is probably one of the most vital elements in Christianity. However, churches nowadays witness two extreme attitudes in approaching the Lord’s Supper: one that over-sacralizes the ceremony as something mystical or magical while the other simply takes it as a ritual or memorial. While both notions are not wrong in some sense, the ceremony in fact falls somewhere in the middle. Eucharist involved two important dimensions: it is a meal, and it is a “sacrificial” meal. The ordinary and the religious aspects both exist within the eucharist. \nIn the records of the Last Supper, Jesus ate the Passover “meal” with His disciples and reinstituted it. From then on, the early church celebrated the Lord’s Supper, gathering and breaking bread in houses, which was then known as agape—a love feast. However, what we witness in today’s Lord’s Supper is nowhere close to the original Last Supper or the early Christian agape feast. It becomes a ceremony without a meal; a celebration without a feast. It is ironic that the so-called “supper” only involves a wafer-like bread and a really small cup of wine. It is the absence of a “meal” that this ceremony becomes more and more detached. The Lord’s Supper becomes difficult to understand because of the emphasis on its sacredness. The ceremony remains a ritual as the “sacred” is separated from the “secular”. \nIt is the contention of this study that the separation of the love feast from the ceremony of the Lord’s Supper that render it meaningless. This study aims to uncover the context and history of the Lord’s Supper, especially the significance of a feast or meal in the eucharist, and how it was lost in the course of history. We will see that it is in the context of a meal that the early church celebrates the eucharist, a thanksgiving in the form of a love feast. It is in the context of a meal that Jesus introduced His body and blood in the Last Supper. It is in the context of a meal that God commanded the Israelites to observe the Passover. When we approach the Lord’s Table without a proper meal, the eating of the “bread” and “wine” without context becomes a ritual without reality. \nFinally, suggestions are given to better approach the Lord’s Table, and hopefully regain the meaning and spirit of the ceremony.",
 'version': 'v2'}
"""
v_ptr = v_list_local[-1]  # s2orc: title \n\n Abstract para1\n\n\nSection header 1\nSection para 1\n\nSection header 2\nSec2 para 1\Sec2 para 2...
# 'text': "Special values of spectral zeta functions of graphs and Dirichlet $L$-functions\n\nIn this paper, we establish relations between special values of Dirichlet $L$-functions and that of spectral zeta functions or $L$-functions of cycle graphs. In fact, they determine each other in a natural way. These two kinds of special values were bridged together by a combinatorial derivative formula obtained from studying spectral zeta functions of the first order self-adjoint differential operators on the unit circle.\n\n\nSpecial values of Riemann zeta function and of Dirichlet L-functions\nIn 1735, Euler proved that the special values of the Riemann zeta function for any n ∈ N ζ(2n) := ∞ k=1 1 k 2n = (−1) n+1 (2π) 2n 2 · (2n)! B 2n ∈ π 2n Q, (1.1) where B k denotes the k-th Bernoulli number, given via the generating function He also showed for a special Dirichlet L-series L(2n + 1, where E 2n is the 2n-th secant number, given by the formula In 1940, Hecke [7] first extended the above results to Dirichlet L-series L(s, χ) := +∞ k=1 χ(k) k s , for s ∈ C with ℜ(s) > 1 and real quadratic Dirichlet character χ : (Z/N Z) × → C × . For a general Dirichlet character, Leopoldt [8] studied the values of L(n, χ) in the case that n and χ satisfying the parity condition (i.e., χ(−1) = (−1) n ) using the generalize Bernoulli numbers in 1958. More precisely, if χ (mod N ) is a primitive character and n a positive integer satisfying the parity condition χ(−1) = (−1) n , then where G(χ) = 1≤a≤N χ(a)e 2πia N is the Gauss sum and B n (x) = 0≤j≤n n j B j x n−j the n-th Bernoulli polynomial.\nNote that formula (1.3) only holds for primitive characters! For a non-primitive character, one needs to lift it to the primitive case to find the special values. In our Theorems 3.1 and 4.2 below, we give new formulae for special values of Dirichlet L-function which work for arbitrary characters.\n\nSpectral zeta functions of cycle graphs Z/NZ\nThe cycle graph C N is a simple graph that consists N vertices connected in a closed chain, which might be visualized as the N -th polygon in a plane. It is one of the most fundamental graphs in graph theory, and is also one of the simplest and the most basic examples of Cayley graphs. Recall, a Cayley graph (G, S) with a given group G and a symmetric generating set S ⊆ G (i.e. S = S −1 ) is the graph which has the elements of G as vertex sets and two vertices g 1 and g 2 are adjacent if there is an element s ∈ S such that g 1 = g 2 s. Note that the cycle graph C N is just the Cayley graph with G = Z/N Z and generating set S = {1, −1}.\nFor the cycle graph C N , its Laplacian operator on L 2 (V ), V = Z/N Z the vertex set, is given by ∆f (x) := 1 4 (2f (x) − f (x + 1) − f (x − 1)), (1.4) for any f ∈ L 2 (V ) and x ∈ Z/N Z. For m ∈ {1, 2, · · · , N − 1}, a direct computation shows that is the eigenvector associated with the eigenvalue sin 2 ( mπ N ). The spectral zeta function of the graph C N is defined as .\nFor any Dirichlet character χ : (Z/N Z) × → C × , we define the Dirichlet L-function for the cycle graph C N as .\nAs an analogue of graph L-function, we also introduce the following notation .\nThe spectral zeta functions for cycle graphs C N and Dirichlet L-functions of cycle graphs are not new and have already appeared in the literature. For example, they are main subjects of studies in the recent interesting papers [2], [6] and [5]. The main points of [6] and [5] are to use ζ Z/N Z (s) and L Z/N Z (s, χ) to approximate the classical Riemann zeta function ζ(s) and Dirichlet L-function L(s, χ), respectively. Especially, they reformulate the (generalized) Riemann Hypothesis in terms of these graph zeta functions or L-functions.\nFriedli and Karlsson emphasize that the spectral zeta function ζ Z/N Z (s) is a more natural approximation of ζ(s) than other artificial ones, see [6,. Here we give another support for the importance of studying these functions from the point of view of special values. Indeed, we find the special values of (1.3) can also be expressed as that of the spectral zeta function of the graph Z/N Z. One of our main results is the following formulae (cf. Theorem 4.2): Theorem A. (i) For any even character χ (mod N ) and any integer n ≥ 1, we have (ii) For any odd character χ (mod N ) and any integer n ≥ 1, we have where a n,i are explicitly given constants which only depend on n and i and b n,i = ia n,i .\nFor the exact combinatorial expression of a n,i , see (4.5) in Section 4. We remark that a n,i are also completely determined by the Laurent expansion of 1 sin 2n x , see Proposition 4.5 for details. As a consequence, we also have similar relations between special values of the Riemann zeta function and of the spectral zeta functions of the cycle graphs (cf. Theorem 4.4) : Theorem B. For any integers N ≥ 2 and n ≥ 1, we have where a n,i is the same as given in Theorem A.\n\nFirst order self-adjoint differential operator\nOur project begins with a study of the spectral zeta function of the first order self-adjoint differential operators on the unit circle S 1 of the form For any s = n ∈ N\\{0}, thanks to Mercer's Theorem (cf. [3, §3.5.4]), we can relate the spectral series in (1.7) with the integral of Green function of the differential operator. For n ≥ 3, the calculation of the integral naturally involves classical combinatorial problems on counting n-permutations.\nUsing these facts, we study the special values of the series and give a new combinatorial interpretation and deduces an explicit formula for this sum. We systematically use the classical Mercer's theorem for a differential operator, which can be viewed as a baby version of the trace formula, to evaluate special values. As a corollary, we prove the above known results in a new simple way. In the particular case α = π/2, we recover the above result of Euler. Here, we only consider the first order differential operator case and will treat the second order operators, i.e., Sturm-Liouville operators in [12].\nSeries (1.8) are closely related to the special values of Hurwitz zeta function and Dirichlet L-functions (cf. [4]), which arise out of number theory problems and other considerations. Using the explicit formula for series (1.8), we deduce special values of Riemann zeta function and Dirichlet L-series, which recover Euler and Hecke's classical results.\n2 Special values of spectral zeta functions of the first order differential operators In this section we study special values of spectral zeta functions of the first order self-adjoint differential operators on the unit circle S 1 of the form (1.5). It turns out its spectrum is quite simple and one can easily write down its spectral zeta function. Then we use the classical Mercer's theorem to evaluate integral values of the spectral zeta function. This naturally leads to the combinatorial problems on counting permutations with fixed difference number between descents and ascents. In subsection 2.1, we recall basic facts of first order differential operators, especially, Green functions and Mercer's theorem. Basic combinatorial facts on counting permutations will be recalled in the next subsection. In the final subsection 2.3, we give the main theorem of this section.\n\nGreen function and Mercer's Theorem\nThe first order self-adjoint differential operators (1.5) is equivalent to the boundary value problem with the boundary condition u(0) = u(1).\nThen T v is a self-adjoint operator with the domain where AC[0, 1] represents all the absolutely continuous functions on [0, 1] and The k-th eigenvalue of problem (2.1) and (2.2) is the same as λ k in (1.6), and the corresponding eigenfunction is Note cλ k is a translation of 2kπ by α = 1 0 vdx, which is related to the integral of the potential function Therefore, without loss of generality, we consider the boundary value problems (2.3) with the boundary condition (2.2). In the following, we always assume that α = 2kπ, k = 0, ±1, ±2, · · · , then 0 is not the eigenvalues of T . Hence T −1 exists and is a bounded linear operator on L 2 [0, 1], and the Green function G(s, t) of (2.3) at 0 is defined as that for any f ∈ L 2 [0, 1], The definition is equivalent to for any fixed s ∈ [0, 1], where δ s (t) is the Delta function at s. Hence the Green function satisfies By the definition, we get the Green function of T at 0, Using eigenfunctions {ϕ k (x) = e 2kπix , k = 0, ±1, ±2, · · · } of the operator T , we get a series representation of G(s, t). In fact, for any fixed s ∈ [0, 1], we can expand G(s, t) by Fourier series where the Fourier coefficient where η := η(s, t) := e i2π(t−s) . Note for any fixed s (or t), due to the discontinuity of the Green function G(s, t), the series (2.6) is not uniformly convergent. We can use the method in [11, §2.1] to deal with this issue. Let's consider the limit For any given ε > 0, we have and the last series uniformly converges. Therefore, Thanks to the uniform convergence, we apply Mercer's Theorem to the first order differential operator (2.3) to get Thus in order to evaluate series (1.8), it is enough to calculate the integral on the right hand side of (2.8). For n = 2, we just notice that Hence, However, to calculate the right hand side of (2.8) for general n ≥ 3, we need to compare the consecutive x j and x j+1 and then get an explicit integrand. More precisely, by noting that where we set x n+1 = x 1 for our convenience, and ♯A means the cardinality of a finite set A. This comparison problem naturally leads to a classical combinatorial problem on counting permutations.\nTherefore, the claim follows.\n\nA combinatorial derivative formula\nGiven (x 1 , · · · , x n ) ∈ [0, 1] n , we may assume x i = x j , for any i = j. Then there exists a unique σ ∈ S n such that Therefore, by (2.10), we have It follows that e −i(n/2−l)α A(n, l).\nTheorem 2.4. For any α = 2kπ, k = 0, ±1, ±2, · · · , and n ≥ 2, we have +∞ k=−∞ 1 (2kπ + α) n = 1 2 n (n − 1)! sin n (α/2) n−1 l=1 cos ((n/2 − l)α) A(n, l), (2.12) and for n = 1, we have the well known formula Remark 2.5. Theorem 2.4 is indeed a (combinatorial) higher order derivative formula for cot x, i.e., Once one has known the expression in (2.12), it would be possible to prove Theorem 2.4 directly by induction, and by using the induction formula of A(n, l)(cf. [1, Theorem 1.7]) and playing with triangle identities. We, however, prefer to keep this seemingly more complicated way to establish the result for two reasons: first, this is the way how we found this expression exactly; second, the idea of using Green function, Mercer's theorem and combinatorics might be useful for higher-order-operator or higher-dimension situations.\nIn particular, let α = π/2 in Theorem 2.4, we get the following identities: for n ≥ 2, if n is odd.\nCorollary 2.6. The 2n-th Bernoulli number For our purpose, the above relation can also be written as Similarly, the spectral zeta function is also a linear combination of Hurwitz zeta functions:  . (4.1) The special values of these spectral zeta functions are well studied in mathematical physics and algebraic geometry, which are called the Verlinde numbers (cf. [13]) In fact, N 2 g V g (N ) is a polynomial in N of degree 3g. More precisely, we have the following explicit computation.\nwhere c g,s is the coefficients of x −2s in the Laurent expansion of (sin x) −2g at x = 0, i.e.\nNote that c g,g = 1 for any integer g ∈ N.\n\n(4.3)\nNote that (4.3) vanishes identically for an odd character. To treat the Dirichlet L-values at odd integers, we also introduce the following notation .\n\nRemark 4.3.\nIt is clear that the coefficients a n,i and b n,i are pure combinatorial and completely independent of the character χ (mod N ). We will show in the Proposition 4.5 that a n,i (hence also b n,i ) are in fact determined by the coefficients c n,i of the Laurent expansion of 1 sin 2n x in Lemma 4.1. We also have a similar formula for ζ(2n) in terms of spectral zeta functions of the cycle graphs.\nTheorem 4.4. For any integers N ≥ 2 and n ≥ 1, we have where a n,i is given in (4.5).\nProof. For each d | N , d > 1, we take the principal character χ 0,d on Z/dZ in Theorem 4.2, we have we obtain d|N ′ d 2n L(2n, χ 0,d ) = π 2n 2(2n − 1)! n i=1 a n,i ζ Z/N Z (i), (4.12) where the sum ′ is taken over all divisors d > 1 of N . Therefore the assertion follows from the following equalities: The above theorem gives us another way to compute the constants {a n,i } in terms of the Laurent expansion coefficients c n,i of 1 sin 2n x defined in Lemma 4.1. Let . . . . . . . . . . . . a n−1,1 a n−1,2 · · · a n−1,n−1 0 a n,1 a n,2 · · · a n,n−1 a n,n c n−1,1 c n−1,2 · · · c n−1,n−1 0 c n,1 c n,2 · · · c n,n−1 c n,n \uf8f6 \uf8f7 \uf8f7 \uf8f7 \uf8f7 \uf8f7 \uf8f8 be the n × n lower triangular matrices given by the constants {a n,i } and {c n,i } respectively. We also let B be the n × n lower triangular matrix Note that B = A · diag{1, 2, · · · , n} since b n,i = ia n,i .\nProposition 4.5. We have (4.14) Proof. Using Euler's formula on even zeta values (1.1) and Lemma 4.1, we have n−s i=0 a n,n−i c n−i,s . (4.16) Since both sides are polynomials in N , by comparing the coefficients of N 2s , we obtain The conclusion follows, since Theorem 4.4 holds for any n.\nAs a consequence, we may also write the special values of L Z/N Z (s, χ) in terms of that of Dirichlet L-function.\nAs an analogue of Zagier's formula for the Verlinder number in Lemma 4.1, we have the following results: Corollary 4.8. (i) For an even primitive character χ (mod N ) and an integer n ≥ 1, we have where B k (x) is the k-th Bernoulli polynomial and G(χ) is the Gauss sum associated toχ.\n\nConcluding remarks\nHere are a few possible interesting topics related to the materials discussed in this paper for further investigations: Evaluation of the cycle graph L-function. In Corollary 4.8, we show, for a primitive character, how to get the special values of L Z/N Z (n, χ) from those of L(n, χ). Note that L Z/N Z (n, χ) is a finite sum of elementary trigonometric functions twisted with a character. It is natural to ask for a direct way to calculate L Z/N Z (n, χ) instead of using the known formulae of Dirichlet L-functions. If so, then we aslo find another way to evaluate special values of Dirichlet L-functions.\nRelations with Iwasawa theory. It is well known that the special values of Dirichlet Lfunctions determine the p-adic L-function by interpolation. Then, given the fact that these values are completely determined by the values of L Z/N Z (s, χ) as in Theorem 4.2, what is the p-adic analogue of L Z/N Z (n, χ) and what is its relation with the p-adic L-function in Iwasawa theory?\nHigher dimensional graph torus. Of course, one can ask similar questions for more general manifolds (and with corresponding graphs) · · · ."

v_ptr = v_list_local[-2]


