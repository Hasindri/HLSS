# export CUDA_VISIBLE_DEVICES="0,1"
export WANDB_PROJECT="HLSS"
export WANDB_NAME="train_hlss_granular"
export WANDB_NOTES="training with text projection using granular descriptions at patch,slide,patient levels with only HVC loss"
export WANDB_RUN_GROUP="Exp016"

wandb on 
# wandb disabled 

python train_hlss_granular.py -c=config/train_hlss_attr3levels.yaml
