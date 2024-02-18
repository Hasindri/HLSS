# export CUDA_VISIBLE_DEVICES="0,1,2"
export WANDB_PROJECT="HLSS"
export WANDB_NAME="train_hlss_KL"
export WANDB_NOTES="2 CLIP resnets, 1 MLP and 3 TExt projs"
export WANDB_RUN_GROUP="Exp020"
# export WANDB_RUN_ID="w8bbvbnl"

# wandb on 
wandb disabled 

python train_hlss_KL.py -c=config/train_hlss_attr3levels.yaml
