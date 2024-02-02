export CUDA_VISIBLE_DEVICES="0"
export WANDB_PROJECT="HLSS"
export WANDB_NAME="eval_hlss_resnet"
export WANDB_NOTES="resnet + HLSS"
export WANDB_RUN_GROUP="Exp015"
# export WANDB_RUN_ID="yd0brywb"

# wandb on 
wandb disabled 

python eval_resnet_hlss.py -c=config/eval_resnet_hlss.yaml
