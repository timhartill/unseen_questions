# Run iterator on fully processed file to output tsv formatted datasets with different context building params than originally

# creak dev
# originally output to $UQA_DIR/creak_fullwiki_bs150_noimplrel/dev.tsv:
#--resume_dir /large_data/thar011/out/mdr/logs/ITER_fullwiki_creakdev_test44_b150_h4_hpqahovnqmubs250_mom-09-06-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue


#    --ctx_gold_sents_only \      # do not prepend/append prior/following sentences to gold sentence
#    --ctx_topk_paras -1          # -1 = include all paras, pos int = include that number max



cd ../code

echo "Creating tsv dataset with all paras but only best sentences..."
python mdr_searchers.py \
    --resume_dir /large_data/thar011/out/mdr/logs/ITER_fullwiki_creakdev_test44_b150_h4_hpqahovnqmubs250_mom-09-06-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue \
    --output_dataset $UQA_DIR/creak_fullwiki_bs150_noimplrel_bestsentsonly/dev.tsv \
    --ctx_gold_sents_only \
    --ctx_topk_paras -1

echo "Creating tsv dataset with max 4 paras and only best sentences..."
python mdr_searchers.py \
    --resume_dir /large_data/thar011/out/mdr/logs/ITER_fullwiki_creakdev_test44_b150_h4_hpqahovnqmubs250_mom-09-06-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue \
    --output_dataset $UQA_DIR/creak_fullwiki_bs150_noimplrel_bestsents_maxp4/dev.tsv \
    --ctx_gold_sents_only \
    --ctx_topk_paras 4

echo "Creating tsv dataset with max 4 paras..."
python mdr_searchers.py \
    --resume_dir /large_data/thar011/out/mdr/logs/ITER_fullwiki_creakdev_test44_b150_h4_hpqahovnqmubs250_mom-09-06-2022-ITER-16False-tkparas150-s1tksents9-s1useparascrTrue-s2tksents5-s2minsentscr0.1-stmaxhops4-stevthresh1.01-stansconf99999.0-rusesentsFalse-rtitlesTrue \
    --output_dataset $UQA_DIR/creak_fullwiki_bs150_noimplrel_maxp4/dev.tsv \
    --ctx_topk_paras 4


