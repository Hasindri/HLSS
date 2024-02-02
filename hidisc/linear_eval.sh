export CUDA_VISIBLE_DEVICES="3"
export WANDB_PROJECT="HLSS"
export WANDB_NAME="train_hlss_patchtxt"
export WANDB_NOTES="training CLIP resnet + text projection on patch level pairs + ML proj on slide & patient level pairs"
export WANDB_RUN_GROUP="Exp014"
# export WANDB_RUN_ID="w8bbvbnl"

# wandb on 
wandb disabled 

python linear_eval_generic.py -c=config/eval_hlss_attr.yaml
