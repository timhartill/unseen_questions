# extract llm generated rationales for  DROP and create llm_expl_...only.tsv versions

cd ../code


echo "BLOOM int8 drop and ood examples..."

python extract_rationale.py \
        --llm_dataset drop_llm_expl \
        --output_dataset drop_llm_expl_only


# TODO after run llm_run_bloom_drop3shotOOD_test72_73_single_drop_int8_cot_and_ansonly.sh
#python extract_rationale.py \
#        --llm_dataset drop_llm_expl_ood \
#        --output_dataset drop_llm_expl_only_ood


echo "SV fp16 drop and ood examples..."

python extract_rationale.py \
        --llm_dataset drop_llm_expl_svfp16 \
        --output_dataset drop_llm_expl_only_svfp16

python extract_rationale.py \
        --llm_dataset drop_llm_expl_svfp16_ood \
        --output_dataset drop_llm_expl_only_svfp16_ood


echo "SV int8 drop and ood examples..."

python extract_rationale.py \
        --llm_dataset drop_llm_expl_svint8 \
        --output_dataset drop_llm_expl_only_svint8

python extract_rationale.py \
        --llm_dataset drop_llm_expl_svint8_ood \
        --output_dataset drop_llm_expl_only_svint8_ood



