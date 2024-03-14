# export CUDA_VISIBLE_DEVICES="0"
export WANDB_PROJECT="HLSS"
export WANDB_NAME="Exp001"
export WANDB_NOTES="hidisc model evaluation"
export WANDB_RUN_GROUP="eval_knn"

# wandb on 
wandb disabled 

python eval_knn_accplot.py -c=config/eval_accplot.yaml
