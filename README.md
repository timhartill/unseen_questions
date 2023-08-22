# Teaching Smaller Language Models To Generalise To Unseen Compositional Questions


Tested on Ubuntu 20.04, Python 3.9

## Create virtual environment:

    bash py39_llm_torch2_install.sh

## Initial Setup

**1:** After cloning this repository, edit .bashrc to export three environmental variables which are necessary to run our code and shell scripts:

    export UQA_DIR=/path/to/base/tsv_formatted_datasets

    export LDATA=/path/to/logs_and_model_checkpoints

    export HDATA=/path/to/misc_other

_UQA_DIR_ is where datasets to train/evaluate our BART-based QA/Reasoning models live. Each dataset is in it's own subdirectory under this and will comprose some or all of dev.tsv. train.tsv and test.tsv.

_LDATA_ is where logs and model checkpoints go. Each model training run will be in it's own subdirectory under LDATA.

_HDATA_ is where miscellaneous other things go including json-formatted training datasets for Iterator models and the Wikipedia corpus. For simplicity the instructions below assume all miscellaneous stuff is in this directory but you can choose to create individual subdirectories under this and adapt the instructions.

 
**2:** Download tsv-formatted datasets: https://drive.google.com/file/d/1QMY9GbwMCvNdsRQh66UCsFKnZzjrXMZ_/view?usp=sharing

Decompress the downloaded file (6.5GB):

    tar -xvf datasets_tsv.tar.gz -C $UQA_DIR

NOTE: The Retrieval-Augmented Training Datasets (_RATD_) can be identifed as the subset with the key __fullwiki_bs_ contained in the dataset name.


**3:** Download our pretrained models: https://drive.google.com/file/d/1wgJvMYKHIxbtbeMX1e0n_rwcSFiES3Tc/view?usp=sharing

Decompress the downloaded file (17.5GB):

    tar -xvf models_tsv.tar.gz -C $LDATA


**4:** If training Iterator sub-models or encoding the Wikipedia corpus for sense retrieval in order to use the Iterator for inference:

4.1: Download training datasets for Iterator (Retriever, Stage 1 Paragraph Reranker and Stage 2 Evidence Set Scorer): https://drive.google.com/file/d/1LAH1c8XmLI2dBUImnSLx36fKxLJAHfVl/view?usp=sharing

Decompress the downloaded file (3.2GB):

    tar -xvf datasets_iterator_reranker.tar.gz -C $HDATA


4.2: Download preprocessed Wikipedia corpus: https://drive.google.com/file/d/143MS2mzom4m2AdkvWM93KH89xjOqfx3A/view?usp=sharing

Decompress the downloaded file (17.3GB):

    tar -xvf wiki20200801.tar.gz -C $HDATA

 
 
More instructions coming soon...
