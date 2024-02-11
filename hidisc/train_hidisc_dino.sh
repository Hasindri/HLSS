export CUDA_VISIBLE_DEVICES="0"
export WANDB_PROJECT="HLSS"
export WANDB_NAME="train_hidisc_dino_resnet50"
export WANDB_NOTES="training resnet50 + DINOHead initliased with Exp001 model on all level pairs, distillation in 128 dim space"
export WANDB_RUN_GROUP="Exp018"
# export WANDB_RUN_ID="w8bbvbnl"

wandb on 
# wandb disabled 

python train_hidisc_dino.py --out_dim 128 --arch "resnet50" --config "config/train_hidisc_dino.yaml" --epochs 300 --batch_size_per_gpu 32 --optimizer sgd --lr 0.03 --weight_decay 1e-4 --weight_decay_end 1e-4 --global_crops_scale 0.14 1 --local_crops_scale 0.05 0.14 --data_path "/data1/dri/hidisc/hidisc/datasets/opensrh" --output_dir "exps/Exp018/e"

