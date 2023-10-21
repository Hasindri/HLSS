export WANDB_PROJECT="HLSS"
export WANDB_NAME="Exp001"
export WANDB_NOTES="SRH strongpatient, hidisc model, seed1, batchsize_10"
export WANDB_RUN_GROUP="train_hidisc"

wandb on 
# wandb disabled 

python train_hidisc_accplot.py 
