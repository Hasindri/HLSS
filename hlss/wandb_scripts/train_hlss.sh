# export CUDA_VISIBLE_DEVICES="0"
export WANDB_PROJECT="HLSS"
export WANDB_NAME="train_hlss"
export WANDB_NOTES="training using a generic set of attributes for all levels/pairs and only HVC loss (non-granular descriptions)"
export WANDB_RUN_GROUP="Exp007"

wandb on 
# wandb disabled 

python train_hlss.py -c=config/train_hlss_attr128.yaml
