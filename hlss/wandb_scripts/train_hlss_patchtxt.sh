# export CUDA_VISIBLE_DEVICES="0"
export WANDB_PROJECT="HLSS"
export WANDB_NAME="train_hlss_patchtxt"
export WANDB_NOTES="hierarchical integration of non-granular text and training using only HVC loss"
export WANDB_RUN_GROUP="Exp014"

wandb on 
# wandb disabled 

python train_hlss_patchtxt.py -c=config/train_hlss_attr128.yaml
