export WANDB_PROJECT="HLSS"
export WANDB_NAME="Exp001"
export WANDB_NOTES="SRH strongpatient, hidisc model, seed1, batchsize_10"
export WANDB_RUN_GROUP="eval_knn"

wandb on 
# wandb disabled 

python eval_knn_accplot.py -c=config/eval_accplot.yaml
