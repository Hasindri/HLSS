export CUDA_VISIBLE_DEVICES="0"
export WANDB_PROJECT="HLSS"
export WANDB_NAME="eval_hlss_dino_RN50_65k"
export WANDB_NOTES="hierarchical distillation on top of Exp007 "
export WANDB_RUN_GROUP="Exp018"
# export WANDB_RUN_ID="yd0brywb"
# export RANK=0
# export WORLD_SIZE=2
# export LOCAL_RANK=0
# wandb on 
wandb disabled 

python eval_hlss_dino.py -c="config/eval_hlss_dino.yaml"
