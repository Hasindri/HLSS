# export CUDA_VISIBLE_DEVICES="0"
export WANDB_PROJECT="HLSS"
export WANDB_NAME="train_hlss_KL"
export WANDB_NOTES="training with total HLSS loss, both hierarchical vision contrastive (HVC) and  hierarchical vision-text alignment (HA)(KL divergence) loss"

wandb on 
# wandb disabled 

python train_hlss_KL.py -c=config/train_hlss_attr3levels.yaml
