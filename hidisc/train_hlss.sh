export CUDA_VISIBLE_DEVICES="0"
export WANDB_PROJECT="HLSS"
export WANDB_NAME="train_hlss_3TEXT"
export WANDB_NOTES="training CLIP resnet + 3 text projections on all level pairs"
export WANDB_RUN_GROUP="Exp009"
# export WANDB_RUN_ID="w8bbvbnl"

wandb on 
# wandb disabled 

python train_hlss_3TEXT.py -c=config/train_hlss_attr128.yaml
