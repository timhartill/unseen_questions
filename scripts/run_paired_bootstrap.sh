#!/bin/bash
#Run paired bootstrap tests

cd ../code

echo "Running Paired Bootstrap tests ..."

echo "STRATEGYQA..."
python paired_bootstrap.py --output_file $LDATA/out/mdr/logs/eval_outputs/s11/paired_bootstrap.txt \
        --gold $UQA_DIR/strategy_qa_bigbench_od_ans/dev.tsv \
        --sys1 $LDATA/out/mdr/logs/UQA_s11_v2_all_g1_qa_g2_numlit_wikissvise_from_s9_v2/dev_strategy_qa_bigbench_od_ans_predictions.json \
        --sys2 $LDATA/out/mdr/logs/UQA_s11_v3_all_g1_qa_g2_numlit_wikissvise_from_s9_v2_addretds_CONTINUE1m/dev_strategy_qa_bigbench_od_ans_predictions.json \
        --eval_type yn \
        --num_samples 10000


python paired_bootstrap.py --output_file $LDATA/out/mdr/logs/eval_outputs/s11/paired_bootstrap.txt \
        --gold $UQA_DIR/strategy_qa_bigbench_fullwiki_bs150_noimplrel/dev.tsv \
        --sys1 $LDATA/out/mdr/logs/UQA_s11_v2_all_g1_qa_g2_numlit_wikissvise_from_s9_v2/dev_strategy_qa_bigbench_fullwiki_bs150_noimplrel_predictions.json \
        --sys2 $LDATA/out/mdr/logs/UQA_s11_v3_all_g1_qa_g2_numlit_wikissvise_from_s9_v2_addretds_CONTINUE1m/dev_strategy_qa_bigbench_fullwiki_bs150_noimplrel_predictions.json \
        --eval_type yn \
        --num_samples 10000


python paired_bootstrap.py --output_file $LDATA/out/mdr/logs/eval_outputs/s11/paired_bootstrap.txt \
        --gold $UQA_DIR/strategy_qa_bigbench_expl_ans/dev.tsv \
        --sys1 $LDATA/out/mdr/logs/UQA_s11_v2_all_g1_qa_g2_numlit_wikissvise_from_s9_v2/dev_strategy_qa_bigbench_expl_ans_predictions.json \
        --sys2 $LDATA/out/mdr/logs/UQA_s11_v3_all_g1_qa_g2_numlit_wikissvise_from_s9_v2_addretds_CONTINUE1m/dev_strategy_qa_bigbench_expl_ans_predictions.json \
        --eval_type yn \
        --num_samples 10000


python paired_bootstrap.py --output_file $LDATA/out/mdr/logs/eval_outputs/s11/paired_bootstrap.txt \
        --gold $UQA_DIR/strategy_qa_bigbench_gold_context_0/dev.tsv \
        --sys1 $LDATA/out/mdr/logs/UQA_s11_v2_all_g1_qa_g2_numlit_wikissvise_from_s9_v2/dev_strategy_qa_bigbench_gold_context_0_predictions.json \
        --sys2 $LDATA/out/mdr/logs/UQA_s11_v3_all_g1_qa_g2_numlit_wikissvise_from_s9_v2_addretds_CONTINUE1m/dev_strategy_qa_bigbench_gold_context_0_predictions.json \
        --eval_type yn \
        --num_samples 10000


python paired_bootstrap.py --output_file $LDATA/out/mdr/logs/eval_outputs/s11/paired_bootstrap.txt \
        --gold $UQA_DIR/strategy_qa_bigbench_gold_context_1/dev.tsv \
        --sys1 $LDATA/out/mdr/logs/UQA_s11_v2_all_g1_qa_g2_numlit_wikissvise_from_s9_v2/dev_strategy_qa_bigbench_gold_context_1_predictions.json \
        --sys2 $LDATA/out/mdr/logs/UQA_s11_v3_all_g1_qa_g2_numlit_wikissvise_from_s9_v2_addretds_CONTINUE1m/dev_strategy_qa_bigbench_gold_context_1_predictions.json \
        --eval_type yn \
        --num_samples 10000


python paired_bootstrap.py --output_file $LDATA/out/mdr/logs/eval_outputs/s11/paired_bootstrap.txt \
        --gold $UQA_DIR/strategy_qa_bigbench_gold_context_2/dev.tsv \
        --sys1 $LDATA/out/mdr/logs/UQA_s11_v2_all_g1_qa_g2_numlit_wikissvise_from_s9_v2/dev_strategy_qa_bigbench_gold_context_2_predictions.json \
        --sys2 $LDATA/out/mdr/logs/UQA_s11_v3_all_g1_qa_g2_numlit_wikissvise_from_s9_v2_addretds_CONTINUE1m/dev_strategy_qa_bigbench_gold_context_2_predictions.json \
        --eval_type yn \
        --num_samples 10000



echo "COMMONSENSEQA..."
python paired_bootstrap.py --output_file $LDATA/out/mdr/logs/eval_outputs/s11/paired_bootstrap.txt \
        --gold $UQA_DIR/commonsenseqa/dev.tsv \
        --sys1 $LDATA/out/mdr/logs/UQA_s11_v2_all_g1_qa_g2_numlit_wikissvise_from_s9_v2/dev_commonsenseqa_predictions.json \
        --sys2 $LDATA/out/mdr/logs/UQA_s11_v3_all_g1_qa_g2_numlit_wikissvise_from_s9_v2_addretds_CONTINUE1m/dev_commonsenseqa_predictions.json \
        --eval_type mc \
        --num_samples 10000


python paired_bootstrap.py --output_file $LDATA/out/mdr/logs/eval_outputs/s11/paired_bootstrap.txt \
        --gold $UQA_DIR/commonsenseqa_fullwiki_bs150_noimplrel/dev.tsv \
        --sys1 $LDATA/out/mdr/logs/UQA_s11_v2_all_g1_qa_g2_numlit_wikissvise_from_s9_v2/dev_commonsenseqa_fullwiki_bs150_noimplrel_predictions.json \
        --sys2 $LDATA/out/mdr/logs/UQA_s11_v3_all_g1_qa_g2_numlit_wikissvise_from_s9_v2_addretds_CONTINUE1m/dev_commonsenseqa_fullwiki_bs150_noimplrel_predictions.json \
        --eval_type mc \
        --num_samples 10000


echo "DROP..."
python paired_bootstrap.py --output_file $LDATA/out/mdr/logs/eval_outputs/s11/paired_bootstrap.txt \
        --gold $UQA_DIR/drop/dev.tsv \
        --sys1 $LDATA/out/mdr/logs/UQA_s11_v2_all_g1_qa_g2_numlit_wikissvise_from_s9_v2/dev_drop_predictions.json \
        --sys2 $LDATA/out/mdr/logs/UQA_s11_v3_all_g1_qa_g2_numlit_wikissvise_from_s9_v2_addretds_CONTINUE1m/dev_drop_predictions.json \
        --eval_type f1 \
        --num_samples 10000


echo "IIRC..."
python paired_bootstrap.py --output_file $LDATA/out/mdr/logs/eval_outputs/s11/paired_bootstrap.txt \
        --gold $UQA_DIR/iirc_initial_context_fullwiki_bs150/dev.tsv \
        --sys1 $LDATA/out/mdr/logs/UQA_s11_v2_all_g1_qa_g2_numlit_wikissvise_from_s9_v2/dev_iirc_initial_context_fullwiki_bs150_predictions.json \
        --sys2 $LDATA/out/mdr/logs/UQA_s11_v3_all_g1_qa_g2_numlit_wikissvise_from_s9_v2_addretds_CONTINUE1m/dev_iirc_initial_context_fullwiki_bs150_predictions.json \
        --eval_type f1 \
        --num_samples 10000


python paired_bootstrap.py --output_file $LDATA/out/mdr/logs/eval_outputs/s11/paired_bootstrap.txt \
        --gold $UQA_DIR/iirc_gold_context/dev.tsv \
        --sys1 $LDATA/out/mdr/logs/UQA_s11_v2_all_g1_qa_g2_numlit_wikissvise_from_s9_v2/dev_iirc_gold_context_predictions.json \
        --sys2 $LDATA/out/mdr/logs/UQA_s11_v3_all_g1_qa_g2_numlit_wikissvise_from_s9_v2_addretds_CONTINUE1m/dev_iirc_gold_context_predictions.json \
        --eval_type f1 \
        --num_samples 10000


echo "ARC-DA..."
python paired_bootstrap.py --output_file $LDATA/out/mdr/logs/eval_outputs/s11/paired_bootstrap.txt \
        --gold $UQA_DIR/arc_da_od_ans/dev.tsv \
        --sys1 $LDATA/out/mdr/logs/UQA_s11_v2_all_g1_qa_g2_numlit_wikissvise_from_s9_v2/dev_arc_da_od_ans_predictions.json \
        --sys2 $LDATA/out/mdr/logs/UQA_s11_v3_all_g1_qa_g2_numlit_wikissvise_from_s9_v2_addretds_CONTINUE1m/dev_arc_da_od_ans_predictions.json \
        --eval_type f1 \
        --num_samples 10000


python paired_bootstrap.py --output_file $LDATA/out/mdr/logs/eval_outputs/s11/paired_bootstrap.txt \
        --gold $UQA_DIR/arc_da_od_ans_fullwiki_bs150/dev.tsv \
        --sys1 $LDATA/out/mdr/logs/UQA_s11_v2_all_g1_qa_g2_numlit_wikissvise_from_s9_v2/dev_arc_da_od_ans_fullwiki_bs150_predictions.json \
        --sys2 $LDATA/out/mdr/logs/UQA_s11_v3_all_g1_qa_g2_numlit_wikissvise_from_s9_v2_addretds_CONTINUE1m/dev_arc_da_od_ans_fullwiki_bs150_predictions.json \
        --eval_type f1 \
        --num_samples 10000


python paired_bootstrap.py --output_file $LDATA/out/mdr/logs/eval_outputs/s11/paired_bootstrap.txt \
        --gold $UQA_DIR/arc_da_expl_ans/dev.tsv \
        --sys1 $LDATA/out/mdr/logs/UQA_s11_v2_all_g1_qa_g2_numlit_wikissvise_from_s9_v2/dev_arc_da_expl_ans_predictions.json \
        --sys2 $LDATA/out/mdr/logs/UQA_s11_v3_all_g1_qa_g2_numlit_wikissvise_from_s9_v2_addretds_CONTINUE1m/dev_arc_da_expl_ans_predictions.json \
        --eval_type f1 \
        --num_samples 10000


echo "MUSIQUE..."
python paired_bootstrap.py --output_file $LDATA/out/mdr/logs/eval_outputs/s11/paired_bootstrap.txt \
        --gold $UQA_DIR/musique_mu_dev_odv2_fullwiki_bs150/dev.tsv \
        --sys1 $LDATA/out/mdr/logs/UQA_s11_v2_all_g1_qa_g2_numlit_wikissvise_from_s9_v2/dev_musique_mu_dev_odv2_fullwiki_bs150_predictions.json \
        --sys2 $LDATA/out/mdr/logs/UQA_s11_v3_all_g1_qa_g2_numlit_wikissvise_from_s9_v2_addretds_CONTINUE1m/dev_musique_mu_dev_odv2_fullwiki_bs150_predictions.json \
        --eval_type f1 \
        --num_samples 10000


python paired_bootstrap.py --output_file $LDATA/out/mdr/logs/eval_outputs/s11/paired_bootstrap.txt \
        --gold $UQA_DIR/musique_mu_dev_parasv2/dev.tsv \
        --sys1 $LDATA/out/mdr/logs/UQA_s11_v2_all_g1_qa_g2_numlit_wikissvise_from_s9_v2/dev_musique_mu_dev_parasv2_predictions.json \
        --sys2 $LDATA/out/mdr/logs/UQA_s11_v3_all_g1_qa_g2_numlit_wikissvise_from_s9_v2_addretds_CONTINUE1m/dev_musique_mu_dev_parasv2_predictions.json \
        --eval_type f1 \
        --num_samples 10000

echo "FINISHED!"


