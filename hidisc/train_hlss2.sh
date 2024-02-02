# export CUDA_VISIBLE_DEVICES="0,1,2"
export WANDB_PROJECT="HLSS"
export WANDB_NAME="train_hlss_slidetxt"
export WANDB_NOTES="training CLIP resnet + text projection on upto slide level pairs + ML proj on patient level pairs"
export WANDB_RUN_GROUP="Exp014"
# export WANDB_RUN_ID="w8bbvbnl"

wandb on 
# wandb disabled 

python train_hlss_patchtxt.py -c=config/train_hlss_attr128.yaml
