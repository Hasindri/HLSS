export WANDB_PROJECT="HLSS"
export WANDB_NAME="Exp002"
export WANDB_NOTES="SRH strongpatient, no distillation, CLIPinit, no CLIP aug"
export WANDB_RUN_GROUP="group of exps - e.g. eval_knn_hlss"

wandb on 
# wandb disabled 

python eval_knn_hlss_accplot.py -c=config/eval_hlss_accplot.yaml
