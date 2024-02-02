export CUDA_VISIBLE_DEVICES="1"
export WANDB_PROJECT="HLSS"
export WANDB_NAME="eval_hlss_3MLP"
export WANDB_NOTES="clip resnet + 3MLPs"
export WANDB_RUN_GROUP="Exp009"

wandb on 
# wandb disabled 

python eval_knn_hlss_accplot2.py -c=config/eval_clipResnet_accplot.yaml
