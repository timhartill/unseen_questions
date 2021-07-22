# Compositional Generalisation project 

To set up this project, after cloning this repository:

(1) Download the preprocessed unifiedqa datasets (instructions at https://github.com/allenai/unifiedqa).

(2) Extract the low similarity versions of the datasets from the tar file in /low_similarity_datasets and add them to the dataset directory structure as separate datasets.

(3) Download the numerical reasoning datasets (instructions at https://github.com/ag1988/injecting_numeracy/tree/master/pre_training). 

(4) Download the MMLU datasets (instructions at https://github.com/hendrycks/test).

(5) Preprocess the numerical reasoning tasks using encode_numerical_literacy.py after updating the input and output directories in the file to your environment.

(6) Preprocess the MMLU tasks using encode_mmlu.py after updating the input and output directories in the file to your environment.

(7) Ensure that all of the above datasets are in a common directory e.g. /data/unifiedqa/dataset1, /data/unifiedqa/dataset2 etc.

(8) De-duplicate evaluation datasets using deduplicate.py after updating the dataset directory in the file.

(9) Aggregate the Math MMLU tasks into a single dataset using aggregate_mmlu_math.py after updating the input directory in the file.

(10) Run the training scripts to train separate models for the UQA and UQA+TDND datasets (runtrain_bart_origV3.sh and runtrain_bart_indivdigitsV7.sh ).

(11) Run the evaluation scripts for these models to create the eval_metrics.json files (runevalall_bartlarge_pick_ckpt150k.sh and runevalall_bartlarge_indivdigits_pick_ckpt150k.sh).

(12) Create the sentence embeddings for all train and eval datasets (runsembeddings_bart_indivdigits_tdnd_V7.sh).

(13) Run the test-train sentence embedding similarity calculations (runsim_for_sembeddings_bart_indivdigits_tdnd_V7.sh).

(14) Create reports comparing results over different model runs using the instructions in eval_metrics.py.

(15) Create reports combining test-train similarity with prediction performance using the instructions in overlap_detector.py.

(16) Create report on prediction performance by answer type (numeric or textual) by editing directories in check_least_similar_answers.py and running it.





