export CUDA_VISIBLE_DEVICES="0"
export WANDB_PROJECT="HLSS"
export WANDB_NAME="train_dino_resnet50_65k"
export WANDB_NOTES="hierarchical distillation from scratch , resnet50 + DINOHead randomly initliased on all level pairs, 65k"
export WANDB_RUN_GROUP="Exp018"
# export WANDB_RUN_ID="w8bbvbnl"

wandb on 
# wandb disabled 

python main_dino.py --arch "resnet50" --config "config/train_dino.yaml" --epochs 300 --batch_size_per_gpu 32 --optimizer sgd --lr 0.03 --weight_decay 1e-4 --weight_decay_end 1e-4 --global_crops_scale 0.14 1 --local_crops_scale 0.05 0.14 --data_path "/data1/dri/hidisc/hidisc/datasets/opensrh" --output_dir "exps/Exp018/a"

