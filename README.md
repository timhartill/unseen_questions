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

 
**2:** Download tsv-formatted datasets: 

Decompress the downloaded file (6.5GB): https://drive.google.com/file/d/1ZRK_n5Dxgf5NAV5XQV0cQnrPZxQpo3gq/view?usp=sharing

    tar -xvf datasets_tsv.tar.gz -C $UQA_DIR

NOTE: The Retrieval-Augmented Training Datasets (_RATD_) can be identifed as the subset with the key __fullwiki_bs_ contained in the dataset name.


**3:** Download our pretrained models: https://drive.google.com/file/d/1wgJvMYKHIxbtbeMX1e0n_rwcSFiES3Tc/view?usp=sharing

Decompress the downloaded file (17.5GB):

    tar -xvf models_tsv.tar.gz -C $LDATA


**4:** If training Iterator sub-models or encoding the Wikipedia corpus for dense retrieval in order to use the Iterator for inference:

4.1: Download training datasets for Iterator (Retriever, Stage 1 Paragraph Reranker and Stage 2 Evidence Set Scorer): https://drive.google.com/file/d/1SuVlsH6hriAaB56MK4-HoOCa1Ok_2sBU/view?usp=sharing

Decompress the downloaded file (3.2GB):

    tar -xvf datasets_iterator_reranker.tar.gz -C $HDATA


4.2: Download preprocessed Wikipedia corpus: https://drive.google.com/file/d/1qGt-2jxGyGmbbGlOeS7DKewwCIWYxMm7/view?usp=sharing

Decompress the downloaded file (7.3GB):

    tar -xvf wiki20200801.tar.gz -C $HDATA

**5:** Only if performing similarity calculations (not in paper). Download extra low similarity datasets: https://drive.google.com/file/d/1jnzXMUvDga9o3toxrSOxpzC2-Q4SI9dE/view?usp=sharing

Decompress the downloaded file (6.2Mb):

    tar -xvf datasets_tsv_sim.tar.gz -C $UQA_DIR


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
        --mixture q_paras_all,q_paras_noanswer_all,q_mc_all,q_mc_paras_all,q_od_all,tt_all,poet_all,synth_num_all,synthetic_textual,enwiki_20200801_selfsvised,q_ret_paras_all,q_ret_paras_maxp4_all
```

Generally to add new datasets to a training mixture, follow the directions in "dataset_attributes.py".

## Iterator: Encoding Wikipedia for Dense Retrieval

To encode the downloaded Wikipedia paragraphs in subdirectory $LDATA/full_wiki_iterator_retriever_v1:

```
python mdr_encode_corpus_nativeamp.py \
    --do_predict \
    --predict_batch_size 500 \
    --model_name roberta-base \
    --predict_file $HDATA/home/thar011/data/beerqa/enwiki-20200801-pages-articles-compgen-withmerges.jsonl \
    --init_checkpoint $LDATA/iterator_retriever/checkpoint_q_best.pt \
    --embed_save_path $LDATA/full_wiki_iterator_retriever_v1 \
    --use_var_versions \
    --fp16 \
    --max_c_len 300 \
    --num_workers 10
```


## Iterator: Inference i.e. Generating a context for a set of Open Domain, Multi-choice or Partially Contextualised Questions

The following is an example of using the Iterator to generate a context for a tsv-formatted dataset file. The results will be written out as a new tsv-formatted dataset into $UQA_DIR as well as stored in jsonl format in output dir $LDATA/ITER_fullwiki_iirc_dev_using_full_wiki_iterator_retriever_v1... 

If using HNSW as below and running for the first time the HNSW index will be built and saved into $LDATA/full_wiki_iterator_retriever_v1. This takes several hours and requires a lot of RAM.

```
python mdr_searchers.py \
    --prefix ITER_fullwiki_iirc_dev_using_full_wiki_iterator_retriever_v1 \
    --output_dir $LDATA \
    --output_dataset $UQA_DIR/iirc_initial_context_fullwiki_bs60_my_version1/dev.tsv \
    --predict_file $UQA_DIR/iirc_initial_context/dev.tsv \
    --index_path $LDATA/full_wiki_iterator_retriever_v1/index.npy \
    --corpus_dict $LDATA/full_wiki_iterator_retriever_v1/id2doc.json \
    --model_name roberta-base \
    --init_checkpoint $LDATA/iterator_retriever/checkpoint_q_best.pt \
    --model_name_stage google/electra-large-discriminator \
    --init_checkpoint_stage1 $LDATA/iterator_stage1_para_reranker/checkpoint_best.pt \
    --init_checkpoint_stage2 $LDATA/iterator_stage2_evidence_set_scorer/checkpoint_best.pt \
    --gpu_model \
    --hnsw \
    --hnsw_buffersize 40000000 \
    --save_index \
    --beam_size 60 \
    --predict_batch_size 160 \
    --query_add_titles \
    --topk 9 \
    --topk_stage2 5 \
    --s1_use_para_score \
    --s1_para_sent_ratio 0.5 \
    --s1_para_sent_ratio_final -1.0 \
    --s2_use_para_score \
    --s2_para_sent_ratio 0.5 \
    --s2_para_sent_ratio_final -1.0 \
    --max_hops 4 \
    --max_q_len 70 \
    --max_q_sp_len 512 \
    --max_c_len 512 \
    --max_ans_len 35 \
    --s2_sp_thresh 0.10 \
    --s2_min_take 2 \
    --stop_ev_thresh 1.01 \
    --stop_ansconfdelta_thresh 99999.0
```


## Iterator: Training the Retriever, Stage 1 Paragraph Reranker and Stage 2 Evidence Set Scorer

To train the Retriever model:

```
python mdr_train_mhop_nativeamp.py \
    --do_train \
    --prefix RETRIEVER_v1 \
    --predict_batch_size 100 \
    --model_name roberta-base \
    --train_batch_size 150 \
    --learning_rate 2e-5 \
    --fp16 \
    --train_file $HDATA/hpqa_hover_nq_mu_train_with_neg_v0.jsonl \
    --predict_file $HDATA/hpqa_hover_nq_mu_dev_with_neg_v0.jsonl \
    --seed 16 \
    --eval-period -1 \
    --max_c_len 300 \
    --max_q_len 70 \
    --max_q_sp_len 512 \
    --shared-encoder \
    --gradient_accumulation_steps 1 \
    --use_var_versions \
    --reduction none \
    --retrieve_loss_multiplier 1.0 \
    --max_hops 4 \
    --num_negs 2 \
    --random_multi_seq \
    --output_dir $LDATA \
    --num_train_epochs 75 \
    --warmup-ratio 0.1
```

To further train the Retriever using momentum. Note --init_retriever must be updated with the actual directory name which begins with RETRIEVER_v1 but is extended with the date etc:

```
python mdr_train_mhop_nativeamp.py \
    --do_train \
    --prefix RETRIEVER_MOM_v2_from_v1 \
    --predict_batch_size 100 \
    --model_name roberta-base \
    --train_batch_size 250 \
    --learning_rate 1e-5 \
    --fp16 \
    --train_file $HDATA/hpqa_hover_nq_mu_train_with_neg_v0.jsonl \
    --predict_file $HDATA/hpqa_hover_nq_mu_dev_with_neg_v0.jsonl \
    --seed 16 \
    --eval-period -1 \
    --max_c_len 300 \
    --max_q_len 70 \
    --max_q_sp_len 512 \
    --shared-encoder \
    --gradient_accumulation_steps 1 \
    --use_var_versions \
    --output_dir $LDATA \
    --momentum \
    --reduction none \
    --retrieve_loss_multiplier 1.0 \
    --max_hops 4 \
    --num_negs 2 \
    --random_multi_seq \
    --k 76800 \
    --m 0.999 \
    --temperature 1 \
    --init_retriever $LDATA/RETRIEVER_v1..../checkpoint_best.pt \
    --output_dir $LDATA/out/mdr/logs \
    --num_train_epochs 75 \
    --warmup-ratio 0.1
```

To train the Stage 1 Paragraph Reranker:

```
python mdr_train_stage1_nativeamp.py \
    --do_train \
    --prefix STAGE1RERANKER_v1 \
    --predict_batch_size 100 \
    --model_name google/electra-large-discriminator \
    --train_batch_size 12 \
    --learning_rate 5e-5 \
    --fp16 \
    --train_file $HDATA/sent_train.jsonl \
    --predict_file $HDATA/sent_dev.jsonl \
    --seed 42 \
    --eval-period 2000 \
    --max_c_len 512 \
    --max_q_len 70 \
    --gradient_accumulation_steps 8 \
    --use-adam \
    --sp-weight 1.0 \
    --output_dir $LDATA \
    --save_prediction stage1_dev_predictions.jsonl \
    --num_train_epochs 7 \
    --sent_score_force_zero \
    --sp_percent_thresh 0.55 \
    --num_workers_dev 10 \
    --debug \
    --warmup-ratio 0.1
```

To train the Stage 2 Evidence Set Scorer:

```
python mdr_train_stage2_nativeamp.py \
    --do_train \
    --prefix STAGE2EVSETSCORER_v1 \
    --predict_batch_size 100 \
    --model_name google/electra-large-discriminator \
    --train_batch_size 12 \
    --learning_rate 5e-5 \
    --fp16 \
    --train_file $HDATA/sent_train.jsonl \
    --predict_file $HDATA/sent_dev.jsonl \
    --seed 42 \
    --eval-period 2000 \
    --max_c_len 512 \
    --max_q_len 70 \
    --gradient_accumulation_steps 8 \
    --use-adam \
    --sp-weight 1.0 \
    --output_dir $LDATA \
    --save_prediction stage2_dev_predictions.jsonl \
    --num_train_epochs 7 \
    --sent_score_force_zero \
    --sp_percent_thresh 1.0 \
    --num_workers_dev 10 \
    --warmup-ratio 0.1
```


## References
If you find this repository useful, please consider giving a star and citing this work:

Hartill, Tim and TAN, Neset and Witbrock, Michael and Riddle, Patricia J [*Teaching Smaller Language Models To Generalise To Unseen Compositional Questions*](https://arxiv.org/abs/2308.00946)

```bibtex
@ARTICLE{Hartill2023-pf,
  title    = "Teaching Smaller Language Models To Generalise To Unseen Compositional Questions",
  author   = "Hartill, Tim and TAN, Neset and Witbrock, Michael and Riddle, Patricia J",
  journal  = "Transactions on Machine Learning Research",
  month    =  aug,
  year     =  2023
}

```

