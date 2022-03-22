# eval retrieve on pretrained model
# Per issue 8: We used a large beam size for the leaderboard submission. We used beam size 200 and top250 passage sequences for answer extraction.

export PYTHONPATH=/data/thar011/gitrepos/multihop_dense_retrieval


python scripts/eval/eval_mhop_retrieval.py \
    data/hotpot/hotpot_qas_val.json \
    data/hotpot_index/wiki_index.npy \
    data/hotpot_index/wiki_id2doc.json \
    models/q_encoder.pt \
    --batch-size 100 \
    --beam-size 1 \
    --topk 1 \
    --shared-encoder \
    --model-name roberta-base \
    --gpu \
    --save-path timtests/hpqa_val_test.json


python scripts/eval/eval_mhop_retrieval.py \
    data/hotpot/hotpot_qas_val.json \
    data/hotpot_index/wiki_index.npy \
    data/hotpot_index/wiki_id2doc.json \
    models/orig_q_encoder_savedagain.pt \
    --batch-size 100 \
    --beam-size 1 \
    --topk 1 \
    --shared-encoder \
    --model-name roberta-base \
    --gpu \
    --save-path timtests/hpqa_val_test_orig_savedagain.json




python scripts/eval/eval_mhop_retrieval.py \
    data/hotpot/hotpot_qas_val.json \
    data/hotpot_index/wiki_index.npy \
    data/hotpot_index/wiki_id2doc.json \
    logs/01-16-2022/tim_-seed16-bsz100-fp16True-lr2e-05-decay0.0-warm0.1-valbsz100-sharedTrue-multi1-schemenone/checkpoint_best.pt \
    --batch-size 100 \
    --beam-size 1 \
    --topk 1 \
    --shared-encoder \
    --model-name roberta-base \
    --gpu \
    --strict \
    --save-path timtests/hpqa_val_test_timstep1best.json


python scripts/eval/eval_mhop_retrieval.py \
    data/hotpot/hotpot_qas_val.json \
    data/hotpot_index/wiki_index.npy \
    data/hotpot_index/wiki_id2doc.json \
    logs/01-16-2022/tim_-seed16-bsz100-fp16True-lr2e-05-decay0.0-warm0.1-valbsz100-sharedTrue-multi1-schemenone/checkpoint_best.pt \
    --batch-size 100 \
    --beam-size 1 \
    --topk 1 \
    --shared-encoder \
    --model-name roberta-base \
    --gpu \
    --strict \
    --exact \
    --save-path timtests/hpqa_val_test_timstep1bestexact.json



python scripts/eval/eval_mhop_retrieval.py \
    data/hotpot/hotpot_qas_val.json \
    data/hotpot_index/wiki_index.npy \
    data/hotpot_index/wiki_id2doc.json \
    logs/01-17-2022/tim_-seed16-bsz100-fp16True-lr1e-05-decay0.0-warm0-valbsz100-m0.999-k76800-t1.0/checkpoint_q_best.pt \
    --batch-size 100 \
    --beam-size 1 \
    --topk 1 \
    --shared-encoder \
    --model-name roberta-base \
    --gpu \
    --strict \
    --exact \
    --save-path timtests/hpqa_val_test_timstep2qbest.json


python scripts/eval/eval_mhop_retrieval.py \
    data/hotpot/hotpot_qas_val.json \
    data/hpqa_tim_momentum/index.npy \
    data/hpqa_tim_momentum/id2doc.json \
    logs/01-17-2022/tim_-seed16-bsz100-fp16True-lr1e-05-decay0.0-warm0-valbsz100-m0.999-k76800-t1.0/checkpoint_q_best.pt \
    --batch-size 100 \
    --beam-size 1 \
    --topk 1 \
    --shared-encoder \
    --model-name roberta-base \
    --gpu \
    --strict \
    --exact \
    --save-path timtests/hpqa_val_test_timstep2qbest_newcorpus.json


python scripts/eval/eval_mhop_retrieval.py \
    data/hotpot/hotpot_qas_val.json \
    data/hpqa_tim_no_momentum/index.npy \
    data/hpqa_tim_no_momentum/id2doc.json \
    logs/01-16-2022/tim_-seed16-bsz100-fp16True-lr2e-05-decay0.0-warm0.1-valbsz100-sharedTrue-multi1-schemenone/checkpoint_best.pt \
    --batch-size 100 \
    --beam-size 1 \
    --topk 1 \
    --shared-encoder \
    --model-name roberta-base \
    --gpu \
    --strict \
    --exact \
    --save-path timtests/hpqa_val_test_timstep2qbest_newcorpus_nomomentum.json

