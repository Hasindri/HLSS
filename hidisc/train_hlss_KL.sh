export CUDA_VISIBLE_DEVICES="0"
export WANDB_PROJECT="HLSS"
export WANDB_NAME="train_hlss_KL"
export WANDB_NOTES="train_hlss_granular setup + enforcing KL div loss between visual feature and text features in 1024"
export WANDB_RUN_GROUP="Exp020"
# export WANDB_RUN_ID="w8bbvbnl"

wandb on 
# wandb disabled 

python train_hlss_KL.py -c=config/train_hlss_attr3levels.yaml
