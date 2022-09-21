# Run iterator on fully processed file to output tsv formatted datasets with different context building params than originally

# creak dev + train


# originally output to $UQA_DIR/creak_fullwiki_bs150_implrel/dev.tsv:  
#/large_data/thar011/out/mdr/logs/ITER_fullwiki_creakdev_test45ir_b150_h4_hpqahovnqmubs250_mom-09-11-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue

# originally output to $UQA_DIR/creak_fullwiki_bs150_implrel/train.tsv:
#   --resume_dir /large_data/thar011/out/mdr/logs/ITER_fullwiki_creaktrain_test48_b150_h4_hpqahovnqmubs250_mom-09-17-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue


#    --ctx_gold_sents_only \      # do not prepend/append prior/following sentences to gold sentence
#    --ctx_topk_paras -1          # -1 = include all paras, pos int = include that number max



cd ../code

echo "Processing CREAK ..."
echo "Creating dev tsv dataset with all paras but only best sentences..."
python mdr_searchers.py \
    --resume_dir /large_data/thar011/out/mdr/logs/ITER_fullwiki_creakdev_test45ir_b150_h4_hpqahovnqmubs250_mom-09-11-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue \
    --output_dataset $UQA_DIR/creak_fullwiki_bs150_implrel_bestsentsonly/dev.tsv \
    --ctx_gold_sents_only \
    --ctx_topk_paras -1

echo "Creating dev tsv dataset with max 4 paras and only best sentences..."
python mdr_searchers.py \
    --resume_dir /large_data/thar011/out/mdr/logs/ITER_fullwiki_creakdev_test45ir_b150_h4_hpqahovnqmubs250_mom-09-11-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue \
    --output_dataset $UQA_DIR/creak_fullwiki_bs150_implrel_bestsents_maxp4/dev.tsv \
    --ctx_gold_sents_only \
    --ctx_topk_paras 4

echo "Creating dev tsv dataset with max 4 paras..."
python mdr_searchers.py \
    --resume_dir /large_data/thar011/out/mdr/logs/ITER_fullwiki_creakdev_test45ir_b150_h4_hpqahovnqmubs250_mom-09-11-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue \
    --output_dataset $UQA_DIR/creak_fullwiki_bs150_implrel_maxp4/dev.tsv \
    --ctx_topk_paras 4


echo "Creating train tsv dataset with all paras but only best sentences..."
python mdr_searchers.py \
    --resume_dir /large_data/thar011/out/mdr/logs/ITER_fullwiki_creaktrain_test48_b150_h4_hpqahovnqmubs250_mom-09-17-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue \
    --output_dataset $UQA_DIR/creak_fullwiki_bs150_implrel_bestsentsonly/train.tsv \
    --ctx_gold_sents_only \
    --ctx_topk_paras -1

echo "Creating train tsv dataset with max 4 paras and only best sentences..."
python mdr_searchers.py \
    --resume_dir /large_data/thar011/out/mdr/logs/ITER_fullwiki_creaktrain_test48_b150_h4_hpqahovnqmubs250_mom-09-17-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue \
    --output_dataset $UQA_DIR/creak_fullwiki_bs150_implrel_bestsents_maxp4/train.tsv \
    --ctx_gold_sents_only \
    --ctx_topk_paras 4

echo "Creating train tsv dataset with max 4 paras..."
python mdr_searchers.py \
    --resume_dir /large_data/thar011/out/mdr/logs/ITER_fullwiki_creaktrain_test48_b150_h4_hpqahovnqmubs250_mom-09-17-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue \
    --output_dataset $UQA_DIR/creak_fullwiki_bs150_implrel_maxp4/train.tsv \
    --ctx_topk_paras 4


python replace_implicit_relation_question_with_original.py \
        --orig_dataset creak_od_ans \
        --iter_dataset creak_fullwiki_bs150_implrel_bestsentsonly \
        --output_dataset creak_fullwiki_bs150_implrel_bestsentsonly_origq


python replace_implicit_relation_question_with_original.py \
        --orig_dataset creak_od_ans \
        --iter_dataset creak_fullwiki_bs150_implrel_bestsents_maxp4 \
        --output_dataset creak_fullwiki_bs150_implrel_bestsents_maxp4_origq


python replace_implicit_relation_question_with_original.py \
        --orig_dataset creak_od_ans \
        --iter_dataset creak_fullwiki_bs150_implrel_maxp4 \
        --output_dataset creak_fullwiki_bs150_implrel_maxp4_origq





