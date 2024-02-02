export CUDA_VISIBLE_DEVICES="1,2"
export WANDB_PROJECT="HLSS"
export WANDB_NAME="train_hidisc_3MLP"
export WANDB_NOTES="train hidisc with 3 unfrozen MLPs, SRH strong patient"
export WANDB_RUN_GROUP="Exp009"
# export WANDB_RUN_ID="42748ghn"

# wandb on 
wandb disabled 

python train_hidisc.py -c=config/train_hidisc_patient.yaml
