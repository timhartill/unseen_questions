# Teaching Smaller Language Models To Generalise To Unseen Compositional Questions


Tested on Ubuntu 20.04, Python 3.9

## Create virtual environment:

    bash py39_llm_torch2_install.sh

## Initial Setup

**1:** After cloning this repository, edit .bashrc to export three environmental variables which are necessary to run our code and shell scripts:

    export UQA_DIR=/path/to/base/tsv_formatted_datasets

    export LDATA=/path/to/logs_and_model_checkpoints

    export HDATA=/path/to/misc_other

_UQA_DIR_ is where the tsv-formatted datasets used to train/evaluate our BART-based QA/Reasoning models live. Each dataset is in it's own subdirectory under this and will comprose some or all of dev.tsv. train.tsv and test.tsv.

_LDATA_ is where logs and model checkpoints go. Each model training run will be in it's own subdirectory under LDATA.

_HDATA_ is where miscellaneous other things go including json-formatted training datasets for Iterator models and the Wikipedia corpus. For simplicity the instructions below assume all miscellaneous stuff is in this directory but you can choose to create individual subdirectories under this and adapt the instructions.

 
**2:** Download tsv-formatted datasets: https://drive.google.com/file/d/1QMY9GbwMCvNdsRQh66UCsFKnZzjrXMZ_/view?usp=sharing

Decompress the downloaded file (6.5GB):

    tar -xvf datasets_tsv.tar.gz -C $UQA_DIR

NOTE: The Retrieval-Augmented Training Datasets (_RATD_) can be identifed as the subset with the key __fullwiki_bs_ contained in the dataset name.


**3:** Download our pretrained models: https://drive.google.com/file/d/1wgJvMYKHIxbtbeMX1e0n_rwcSFiES3Tc/view?usp=sharing

Decompress the downloaded file (17.5GB):

    tar -xvf models_tsv.tar.gz -C $LDATA


**4:** If training Iterator sub-models or encoding the Wikipedia corpus for dense retrieval in order to use the Iterator for inference:

4.1: Download training datasets for Iterator (Retriever, Stage 1 Paragraph Reranker and Stage 2 Evidence Set Scorer): https://drive.google.com/file/d/1LAH1c8XmLI2dBUImnSLx36fKxLJAHfVl/view?usp=sharing

Decompress the downloaded file (3.2GB):

    tar -xvf datasets_iterator_reranker.tar.gz -C $HDATA


4.2: Download preprocessed Wikipedia corpus: https://drive.google.com/file/d/143MS2mzom4m2AdkvWM93KH89xjOqfx3A/view?usp=sharing

Decompress the downloaded file (17.3GB):

    tar -xvf wiki20200801.tar.gz -C $HDATA

## Running Evaluation for Unseen Datasets on BART QA/Reasoning Models 

Run everything below from the code subdirectory. 

### Output Metrics for each Model

Run evaluation for the _Base_ model. This will create a file in the $LDATA/base subdirectory called eval_metrics.json containing metrics for all unseen evaluation datasets plus a selection of the training datasets. A separate json file of predictions for each dataset will also be output:

```
python cli.py --output_dir $LDATA/base \
        --predict_file $UQA_DIR/dev.tsv \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --verbose \
        --model facebook/bart-large \
        --indiv_digits \
        --dont_pretokenize \
        --dont_save_train_token_file \
        --max_output_length 130 \
        --do_predict_all --calc_metrics_all --add_only_missing
```

Next run evaluation for the Base+RATD model by running the same command as above but substituting _base_plus_ratd_ for _base_.

## Training a QA/Reasoning Model


## Iterator: Encoding Wikipedia for Dense Retrieval

## Iterator: Training the Retriever, Stage 1 Paragraph Reranker or Stage 2 Evidence Set Scorer


## Iterator: Inference i.e. Generating a context for a set of Open Domain, Multi-choice or Partially Contextualised Questions




## References
If you find this repository useful, please consider giving a star and citing this work:

[1] Hartill, Tim and TAN, Neset and Witbrock, Michael and Riddle, Patricia J [*Teaching Smaller Language Models To Generalise To Unseen Compositional Questions*](https://arxiv.org/abs/2308.00946)

```bibtex
@ARTICLE{Hartill2023-pf,
  title    = "Teaching Smaller Language Models To Generalise To Unseen Compositional Questions",
  author   = "Hartill, Tim and TAN, Neset and Witbrock, Michael and Riddle, Patricia J",
  journal  = "Transactions on Machine Learning Research",
  month    =  aug,
  year     =  2023
}

```

