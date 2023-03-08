set -x -e
cuda=1
data=single
LR=2e-5
CLIP_LR=2e-7
SPLIT=1
TTYPE=robertabase


CUDA_VISIBLE_DEVICES=$cuda python mlmsa/bert_fine_tuning.py \
    --vtype clip \
    --ttype $TTYPE \
    --mvsa $data \
    --ht \
    --lr $LR \
    --split $SPLIT \
    --freeze_param