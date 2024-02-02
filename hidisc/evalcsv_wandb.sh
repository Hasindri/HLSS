export CUDA_VISIBLE_DEVICES="3"
export WANDB_PROJECT="HLSS"
export WANDB_NAME="eval_resnet_hlss"
export WANDB_NOTES="normal resnet + text projection on all pairs"
export WANDB_RUN_GROUP="Exp015"
# export WANDB_RUN_ID="yd0brywb"

wandb on 
# wandb disabled 

python evalcsv_wandb.py 
