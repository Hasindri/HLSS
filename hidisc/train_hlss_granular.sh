export CUDA_VISIBLE_DEVICES="0,1"
export WANDB_PROJECT="HLSS"
export WANDB_NAME="train_hlss_granular"
export WANDB_NOTES="training CLIP resnet + text projection using 3 granularity descriptions"
export WANDB_RUN_GROUP="Exp016"
# export WANDB_RUN_ID="w8bbvbnl"

wandb on 
# wandb disabled 

python train_hlss_granular.py -c=config/train_hlss_attr3levels.yaml
