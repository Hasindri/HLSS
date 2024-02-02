export CUDA_VISIBLE_DEVICES="2"
export WANDB_PROJECT="HLSS"
export WANDB_NAME="eval_knn_clip"
export WANDB_NOTES="zeroshot KNN evaluation of CLIP resnet on SRH"
export WANDB_RUN_GROUP="Exp011"

wandb on 
# wandb disabled 

python eval_knn.py -c=config/eval_hlss.yaml
