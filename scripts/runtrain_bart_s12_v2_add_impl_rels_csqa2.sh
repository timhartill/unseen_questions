# Add implicit relations to questions and save as new dataset

cd ../code

python add_implicit_relations_to_dataset.py \
        --predict_dataset csqa2 \
        --output_dataset csqa2_impl_rels \
        --model facebook/bart-large \
        --checkpoint $LDATA/out/mdr/logs/UQA_s10_v2_implicitrelations_dsfix_ws50/best-model.pt \
        --predict_batch_size 350 \
        --append_another_bos --do_lowercase \
        --max_output_length 130
        

