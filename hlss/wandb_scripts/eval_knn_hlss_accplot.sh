# export CUDA_VISIBLE_DEVICES="0"
export WANDB_PROJECT="HLSS"
export WANDB_NAME="eval_hlss_KL"
export WANDB_NOTES="evaluation of HLSS model"
export WANDB_RUN_GROUP="Exp020"

# wandb on 
wandb disabled 

python eval_knn_hlss_accplot.py -c=config/eval_hlss_attr128_accplot.yaml
