import re
import wandb
import matplotlib.pyplot as plt

# Initialize lists to store the data
train_patient_loss = []
train_slide_loss = []
train_patch_loss = []
train_sum_loss = []
val_patient_loss = []
val_slide_loss = []
val_patch_loss = []
val_sum_loss = []
epoch = 0

# Regular expressions to match the relevant lines
train_loss_pattern1 = re.compile(r"train/patch_loss_manualepoch (\d+\.\d+)")
train_loss_pattern2 = re.compile(r"train/slide_loss_manualepoch (\d+\.\d+)")
train_loss_pattern3 = re.compile(r"train/patient_loss_manualepoch (\d+\.\d+)")
train_loss_pattern4 = re.compile(r"train/sum_loss_manualepoch (\d+\.\d+)")
val_loss_pattern1 = re.compile(r"val/patch_loss_manualepoch (\d+\.\d+)")
val_loss_pattern2 = re.compile(r"val/slide_loss_manualepoch (\d+\.\d+)")
val_loss_pattern3 = re.compile(r"val/patient_loss_manualepoch (\d+\.\d+)")
val_loss_pattern4 = re.compile(r"val/sum_loss_manualepoch (\d+\.\d+)")

# Initialize Weights and Biases (wandb)
wandb.init(project="HLSS")

# Read the train.log file
train_log = "/data1/dri/hidisc/hidisc/datasets/opensrh/hidisc_strongaug/patient/46de6893-Sep26-10-03-35-patient_disc_dev_/train.log"
with open(train_log, "r") as log_file:
    for line in log_file:
        # Match and extract training loss
        
        train_match1 = train_loss_pattern1.search(line)
        train_match2 = train_loss_pattern2.search(line)
        train_match3 = train_loss_pattern3.search(line)
        train_match4 = train_loss_pattern4.search(line)
        val_match1 = val_loss_pattern1.search(line)
        val_match2 = val_loss_pattern2.search(line)
        val_match3 = val_loss_pattern3.search(line)
        val_match4 = val_loss_pattern4.search(line)

        if train_match1:
            train_patch_loss.append(float(train_match1.group(1)))
            wandb.log({"train/patch_loss": float(train_match1.group(1)), "epoch":epoch})
            
        elif train_match2:
            train_slide_loss.append(float(train_match2.group(1)))
            wandb.log({"train/slide_loss": float(train_match2.group(1)), "epoch":epoch})
            
        elif train_match3:
            train_patient_loss.append(float(train_match3.group(1)))
            wandb.log({"train/patient_loss": float(train_match3.group(1)), "epoch":epoch})
            
        elif train_match4:
            epoch += 1
            train_sum_loss.append(float(train_match4.group(1)))
            wandb.log({"train/sum_loss": float(train_match4.group(1)), "epoch":epoch})

        elif val_match1:
            val_patch_loss.append(float(val_match1.group(1)))
            wandb.log({"val/patch_loss": float(val_match1.group(1)), "epoch":epoch})
            
        elif val_match2:
            val_slide_loss.append(float(val_match2.group(1)))
            wandb.log({"val/slide_loss": float(val_match2.group(1)), "epoch":epoch})
            
        elif val_match3:
            val_patient_loss.append(float(val_match3.group(1)))
            wandb.log({"val/patient_loss": float(val_match3.group(1)), "epoch":epoch})
            
        elif val_match4:
            val_sum_loss.append(float(val_match4.group(1)))
            wandb.log({"val/sum_loss": float(val_match4.group(1)), "epoch":epoch})



epochs = list(range(0, epoch + 1, 1))
# print(f'epoch {epoch}')
# print(f' {len(val_sum_loss)}')

wandb.finish()
