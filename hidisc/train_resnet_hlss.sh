# export CUDA_VISIBLE_DEVICES="3"
export WANDB_PROJECT="HLSS"
export WANDB_NAME="train_resnet_hlss"
export WANDB_NOTES="training  resnet + text projection"
export WANDB_RUN_GROUP="Exp015"
# export WANDB_RUN_ID="w8bbvbnl"

wandb on 
# wandb disabled 

python train_resnet_hlss.py -c=config/train_resnet_hlss_attr128.yaml
