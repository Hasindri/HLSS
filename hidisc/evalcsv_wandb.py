import pandas as pd
import ast
import wandb
import torch

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv('exp15a_eval.csv')
for col in df.columns[1:]:
    for i in range (df.shape[0]):
        x = float(df[col][i].split('(')[-1].split(')')[0])
        df.at[i, col] = x
# breakpoint()

df1 = df.groupby('epoch')['patch'].mean().reset_index()
df2 = df.groupby('epoch')['slide'].mean().reset_index()
df3 = df.groupby('epoch')['patient'].mean().reset_index()
# breakpoint()
# Initiate WandB project
wandb.init(project='HLSS')

# Log the average accuracies for each epoch
for i in range (df1.shape[0]):
    # wandb.log({f'eval_knn/{metric}_acc': wandb.Table(dataframe=df[['epoch', metric]])})

    wandb.log({f"eval_knn/patch_acc": df1['patch'][i]})
    wandb.log({f"eval_knn/slide_acc": df2['slide'][i]})
    wandb.log({f"eval_knn/patient_acc": df3['patient'][i]})

# Finish the run
wandb.finish()
