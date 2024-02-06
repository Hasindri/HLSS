export CUDA_VISIBLE_DEVICES="0,1"
export WANDB_PROJECT="HLSS"
export WANDB_NAME="train_hlss_dino_RN50_65k"
export WANDB_NOTES="training RN50 + DINOHead initliased with Exp007 model on all level pairs"
export WANDB_RUN_GROUP="Exp018"
# export WANDB_RUN_ID="w8bbvbnl"

# wandb on 
wandb disabled 

python train_hlss_dino.py --arch "RN50" --config "config/train_hlss_dino.yaml" --epochs 300 --batch_size_per_gpu 32 --optimizer sgd --lr 0.03 --weight_decay 1e-4 --weight_decay_end 1e-4 --global_crops_scale 0.14 1 --local_crops_scale 0.05 0.14 --data_path "/data1/dri/hidisc/hidisc/datasets/opensrh/train" --output_dir "exps/Exp018/b"

