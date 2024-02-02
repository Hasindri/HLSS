export CUDA_VISIBLE_DEVICES="2"
export WANDB_PROJECT="HLSS"
export WANDB_NAME="train_hlss_attr128"
export WANDB_NOTES="attribute plot"
export WANDB_RUN_GROUP="Exp012"

# wandb on 
wandb disabled 

python attr128_clsmean.py -c=config/eval_hlss_attr.yaml
