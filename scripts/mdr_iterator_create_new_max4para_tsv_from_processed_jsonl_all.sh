# Run iterator on fully processed file to output tsv formatted datasets with different context building params than originally (max 4 paras but +/- 1 sent to gold

# NOTE: creak & csqa2 dev retrieved at beams 150 but train retrieved at beams 60. Also both of these are used WITHOUT implicit relations


#    --ctx_gold_sents_only \      # do not prepend/append prior/following sentences to gold sentence
#    --ctx_topk_paras -1          # -1 = include all paras, pos int = include that number max



cd ../code

echo "Creating CREAK dev tsv dataset with max 4 paras..."
python mdr_searchers.py \
    --resume_dir /large_data/thar011/out/mdr/logs/ITER_fullwiki_creakdev_test44_b150_h4_hpqahovnqmubs250_mom-09-06-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue \
    --output_dataset $UQA_DIR/creak_fullwiki_bs150_noimplrel_maxp4/dev.tsv \
    --ctx_topk_paras 4


echo "Creating CREAK train tsv dataset with max 4 paras..."
python mdr_searchers.py \
    --resume_dir /large_data/thar011/out/mdr/logs/ITER_fullwiki_creaktrain_test62_b60_h4_hpqahovnqmubs250_mom-09-22-2022-ITER-16False-tkparas60-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue \
    --output_dataset $UQA_DIR/creak_fullwiki_bs150_noimplrel_maxp4/train.tsv \
    --ctx_topk_paras 4


echo "Creating CSQA2 dev tsv dataset with max 4 paras..."
python mdr_searchers.py \
    --resume_dir /large_data/thar011/out/mdr/logs/ITER_fullwiki_csqa2dev_test43_b150_h4_hpqahovnqmubs250_mom-09-06-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue \
    --output_dataset $UQA_DIR/csqa2_fullwiki_bs150_noimplrel_maxp4/dev.tsv \
    --ctx_topk_paras 4


echo "Creating CSQA2 train tsv dataset with max 4 paras..."
python mdr_searchers.py \
    --resume_dir /large_data/thar011/out/mdr/logs/ITER_fullwiki_csqa2train_test63_b60_h4_hpqahovnqmubs250_mom-09-22-2022-ITER-16False-tkparas60-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue \
    --output_dataset $UQA_DIR/csqa2_fullwiki_bs150_noimplrel_maxp4/train.tsv \
    --ctx_topk_paras 4


echo "Creating HOVER dev tsv dataset with max 4 paras..."
python mdr_searchers.py \
    --resume_dir /large_data/thar011/out/mdr/logs/ITER_fullwiki_hoverdev_test58_b60_h4_hpqahovnqmubs250_mom-09-20-2022-ITER-16False-tkparas60-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue \
    --output_dataset $UQA_DIR/hover_fullwiki_bs60_maxp4/dev.tsv \
    --ctx_topk_paras 4


echo "Creating HOVER train tsv dataset with max 4 paras..."
python mdr_searchers.py \
    --resume_dir /large_data/thar011/out/mdr/logs/ITER_fullwiki_hovertrain_test59_b60_h4_hpqahovnqmubs250_mom-09-20-2022-ITER-16False-tkparas60-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue \
    --output_dataset $UQA_DIR/hover_fullwiki_bs60_maxp4/train.tsv \
    --ctx_topk_paras 4


echo "Creating QASC dev tsv dataset with max 4 paras..."
python mdr_searchers.py \
    --resume_dir /large_data/thar011/out/mdr/logs/ITER_fullwiki_qascdev_test60_b60_h4_hpqahovnqmubs250_mom-09-20-2022-ITER-16False-tkparas60-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue \
    --output_dataset $UQA_DIR/qasc_fullwiki_bs60_maxp4/dev.tsv \
    --ctx_topk_paras 4


echo "Creating QASC train tsv dataset with max 4 paras..."
python mdr_searchers.py \
    --resume_dir /large_data/thar011/out/mdr/logs/ITER_fullwiki_qasctrain_test61_b60_h4_hpqahovnqmubs250_mom-09-21-2022-ITER-16False-tkparas60-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue \
    --output_dataset $UQA_DIR/qasc_fullwiki_bs60_maxp4/train.tsv \
    --ctx_topk_paras 4


echo "Creating HPQA dev tsv dataset with max 4 paras..."
python mdr_searchers.py \
    --resume_dir /large_data/thar011/out/mdr/logs/ITER_fullwiki_hpqadev_test53_b60_h4_hpqahovnqmubs250_mom-09-19-2022-ITER-16False-tkparas60-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue \
    --output_dataset $UQA_DIR/hpqa_fullwiki_bs60_maxp4/dev.tsv \
    --ctx_topk_paras 4


echo "Creating HPQA train tsv dataset with max 4 paras..."
python mdr_searchers.py \
    --resume_dir /large_data/thar011/out/mdr/logs/ITER_fullwiki_hpqatrain_test52_b60_h4_hpqahovnqmubs250_mom-09-19-2022-ITER-16False-tkparas60-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue \
    --output_dataset $UQA_DIR/hpqa_fullwiki_bs60_maxp4/train.tsv \
    --ctx_topk_paras 4


echo "Creating Musique dev tsv dataset with max 4 paras..."
python mdr_searchers.py \
    --resume_dir /large_data/thar011/out/mdr/logs/ITER_fullwiki_mudev_test55_b60_h4_hpqahovnqmubs250_mom-09-19-2022-ITER-16False-tkparas60-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue \
    --output_dataset $UQA_DIR/musique_qa_fullwiki_bs60_maxp4/dev.tsv \
    --ctx_topk_paras 4


echo "Creating Musique train tsv dataset with max 4 paras..."
python mdr_searchers.py \
    --resume_dir /large_data/thar011/out/mdr/logs/ITER_fullwiki_mutrain_test54_b60_h4_hpqahovnqmubs250_mom-09-19-2022-ITER-16False-tkparas60-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue \
    --output_dataset $UQA_DIR/musique_qa_fullwiki_bs60_maxp4/train.tsv \
    --ctx_topk_paras 4


echo "Creating NQ OPEN dev tsv dataset with max 4 paras..."
python mdr_searchers.py \
    --resume_dir /large_data/thar011/out/mdr/logs/ITER_fullwiki_nqopendev_test51_b60_h4_hpqahovnqmubs250_mom-09-17-2022-ITER-16False-tkparas60-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue \
    --output_dataset $UQA_DIR/nq_open_fullwiki_bs60_maxp4/dev.tsv \
    --ctx_topk_paras 4


echo "Creating NQ OPEN train tsv dataset with max 4 paras..."
python mdr_searchers.py \
    --resume_dir /large_data/thar011/out/mdr/logs/ITER_fullwiki_nqopentrain_test50_b60_h4_hpqahovnqmubs250_mom-09-17-2022-ITER-16False-tkparas60-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue \
    --output_dataset $UQA_DIR/nq_open_fullwiki_bs60_maxp4/train.tsv \
    --ctx_topk_paras 4



