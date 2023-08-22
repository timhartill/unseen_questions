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

## Running Evaluation On BART QA/Reasoning Models 

Run everything below from the /code subdirectory. 

### Output Metrics For Each Model

Run evaluation for the _Base_ model. This will create a file in the $LDATA/base subdirectory called eval_metrics.json containing metrics for all unseen evaluation datasets plus a selection of the training datasets. A separate json file of predictions for each dataset will also be output. If you subsequently run this command again after adding new datasets, inference will only run for the new datasets: and it will be much faster.

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

Next run evaluation for the Base+RATD model by running the same command as above but substituting _base_plus_ratd_ for _base_ or other models you may have trained.

### Create Tables Of Metrics Over A Set Of Models

Running the following will output text files into the _$LDATA/eval_outputs_ subdirectory summarising metrics for the _Base_ versus _Base+RATD_ models. The file with name containing "unseen4" will be for unseen evaluation datasets and that for "seen1" will be for training datasets.

```
python eval_metrics.py --eval_set base_ratd
```

Generally, to add/remove datasets for evaluation and/or to create a new or modified set of models to output a summary for, please refer to the instructions at the top of "dataset_attributes.py".

## Training a QA/Reasoning Model

To train a _Base_ model starting from our checkpoint from the first stage of QA model training:

```
python cli.py --do_train --output_dir $LDATA/my_new_base_model \
        --is_unifiedqa \
        --train_file $UQA_DIR/train.tsv \
        --predict_file $UQA_DIR/dev.tsv \
        --checkpoint $LDATA/stage_1_bart/best-model.pt \
        --train_batch_size 32 \
        --predict_batch_size 32 \
        --append_another_bos --do_lowercase \
        --eval_period 10000 --verbose \
        --gradient_accumulation_steps 4 \
        --wait_step 10 \
        --num_scheduler_steps 250000 \
        --learning_rate 2e-5 \
        --model facebook/bart-large \
        --seed 42 \
        --ssm_prob 0.65 \
        --add_mask_char NONE \
        --max_output_length 130 \
        --fp16 \
        --dont_pretokenize \
        --dont_save_train_token_file \
        --indiv_digits \
        --approx_dev_samples 1250 \
        --g2_prob 0.2 \
        --error_based_ssvise_prob 0.05 \
        --g1_type err \
        --g2_type uni \
        --g2_datasets q_od_all,tt_all,poet_all,synth_num_all,synthetic_textual,enwiki_20200801_selfsvised \
        --mixture q_paras_all,q_paras_noanswer_all,q_mc_all,q_mc_paras_all,q_od_all,tt_all,poet_all,synth_num_all,synthetic_textual,enwiki_20200801_selfsvised
```

To train a _Base+RATD_ model is as above but with --mixture:

```
        --mixture q_paras_all,q_paras_noanswer_all,q_mc_all,q_mc_paras_all,q_od_all,tt_all,poet_all,synth_num_all,synthetic_textual,enwiki_20200801_selfsvised,q_ret_paras_all,q_ret_paras_maxp4_all```

Generally to add new datasets to a training mixture, follow the directions in "dataset_attributes.py".

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

