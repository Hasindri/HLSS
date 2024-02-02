export CUDA_VISIBLE_DEVICES="0"
export WANDB_PROJECT="HLSS"
export WANDB_NAME="Exp001_2000"
export WANDB_NOTES="hidisc model with random resnet + MLP on seed 2000 + epoch 5600 onwards"
export WANDB_RUN_GROUP="eval_knn"
# export WANDB_RUN_ID="x7cprzuc"

wandb on 
# wandb disabled 

python eval_knn_accplot.py -c=config/eval_accplot.yaml
