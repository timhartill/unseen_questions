#!/bin/bash
# add information for incremental eval datasets to existing eval_metrics.json files
# See old run_add_missing_eval_to_existing_pred_calcmetrics.sh for script doing same thing for historical models...


# Note 1: Add new output dirs to appropriate section after running the eval all script for it to ensure the output dir is kept updated with new eval datasets..
# Note 2: Run this from scripts subdirectory..




echo "Update existing BART outputs which use best-model without a particular checkpoint and indiv_digits..."

for out in "${LDATA}/out/mdr/logs/UQA_s11_v1_all_g1_qa_g2_numlit_wikissvise_COPY_AT810Ksteps" "${LDATA}/out/mdr/logs/UQA_s11_v1_all_g1_qa_g2_numlit_wikissvise" "${LDATA}/out/mdr/logs/UQA_s11_v4_all_g1_qa_g2_numlit_wikissvise_addretds" "${LDATA}/out/mdr/logs/UQA_s11_v2_all_g1_qa_g2_numlit_wikissvise_from_s9_v2_COPY_AT810Ksteps" "${LDATA}/out/mdr/logs/UQA_s11_v2_all_g1_qa_g2_numlit_wikissvise_from_s9_v2" "${LDATA}/out/mdr/logs/UQA_s11_v3_all_g1_qa_g2_numlit_wikissvise_from_s9_v2_addretds" "${LDATA}/out/mdr/logs/UQA_s11_v3_all_g1_qa_g2_numlit_wikissvise_from_s9_v2_addretds_COPY_AT_770Ksteps" "${LDATA}/out/mdr/logs/UQA_s11_v3_all_g1_qa_g2_numlit_wikissvise_from_s9_v2_addretds_CONTINUE1m" "${LDATA}/out/mdr/logs/UQA_s11_v6_all_g1_qa_g2_numlit_wikissvise_from_s9_v2_addretdsv2"
do
    echo "Updating eval for $out ..."
    bash runevalall_v2_bartlarge_pick_bestmodel_indivdigits.sh $out
done


echo "Update existing t5-large outputs which use best-model without a particular checkpoint and indiv_digits..."

for out in "${LDATA}/out/mdr/logs/UQA_s11_v5_all_g1_qa_g2_numlit_wikissvise_addretdst5l_COPY_AT560Ksteps" "${LDATA}/out/mdr/logs/UQA_s11_v5_all_g1_qa_g2_numlit_wikissvise_addretdst5l"
do
    echo "Updating eval for $out ..."
    bash runevalall_v2_t5large_pick_bestmodel_indivdigits.sh $out
done



echo Finished!

