export CUDA_VISIBLE_DEVICES="0"
export WANDB_PROJECT="HLSS"
export WANDB_NAME="eval_hlss_granular"
export WANDB_NOTES="hlss + different text projections on different levels "
export WANDB_RUN_GROUP="Exp016"
# export WANDB_RUN_ID="yd0brywb"

# wandb on 
wandb disabled 

python eval_knn_hlss_accplot.py -c=config/eval_hlss_attr128_accplot.yaml
