export WANDB_PROJECT="HLSS"
export WANDB_NAME="train_hlss_attr"
export WANDB_NOTES="visual attribute projection space"
export WANDB_RUN_GROUP="Exp006"

wandb on 
# wandb disabled 

python train_hlss.py -c=config/train_hlss_attr.yaml
