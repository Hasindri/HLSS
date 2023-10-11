import torch
print(torch.__version__)

import tifffile

import os
import numpy as np
from tqdm import tqdm
import yaml

import tarfile

from datasets.srh_dataset import OpenSRHDataset, HiDiscDataset, process_read_im, get_srh_base_aug
from common import (setup_output_dirs, parse_args, get_exp_name,
                           config_loggers, get_optimizer_func,
                           get_scheduler_func, get_dataloaders,get_num_worker)
from datasets.improc import process_read_im, get_srh_base_aug

import warnings


# Ignore all warnings
warnings.filterwarnings("ignore")
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# file_path = "/l/users/hasindri.watawana/hidisc/datasets/opensrh/studies/NIO_097/2/patches/NIO_097-2-2000_2000_600_300.tif"

# if os.path.exists(file_path):
#     print("The file exists.")
# else:
#     print("The file does not exist.")

# patch = torch.from_numpy(tifffile.imread(file_path).astype(np.float32)).contiguous()
# print(f'patch {patch.shape}')

# train_dset = HiDiscDataset(
#         data_root="/l/users/hasindri.watawana/hidisc/datasets/opensrh",
#         studies="train")

# patient0 = train_dset[0]

# print(f"num devices: {torch.cuda.device_count()}")
# print(f"num workers in dataloader: {get_num_worker()}")


# # curr_inst = "/l/users/hasindri.watawana/hidisc/datasets/opensrh/studies/NIO_003/3/patches/NIO_003-3-1000_1000_300_300.tif"
# # print(f'curr_inst {curr_inst}')
# # im = process_read_im(curr_inst) 
# # print(im.shape)

# cf_fd = parse_args()
# # print (f'args {cf_fd}')
# cf = yaml.load(cf_fd, Loader=yaml.FullLoader)
# train_loader, valid_loader = get_dataloaders(cf)  # Get your data loaders here

# # Iterate through the data loader for a few steps
# for batch_idx, batch in enumerate(train_loader):
#     images, labels = batch['image'], batch['label']




# def get_size_in_gb(size_bytes):
#     return size_bytes / (1024 * 1024 * 1024)

# def get_directory_size(directory):
#     total_size = 0
#     for dirpath, dirnames, filenames in os.walk(directory):
#         for filename in filenames:
#             filepath = os.path.join(dirpath, filename)
#             total_size += os.path.getsize(filepath)
#     return total_size

# def main():
#     target_directory = dest_folder
#     total_size_bytes = get_directory_size(target_directory)
#     total_size_gb = get_size_in_gb(total_size_bytes)
#     print(f"Total size of {target_directory}: {total_size_gb:.2f} GB")

# if __name__ == "__main__":
#     main()

# import re

# pattern = r'^(TCGA-\d{2}-\d{4})-\d{2}[A-Z]-\d{2}-DX(\d)\.[A-F0-9-]+\.[a-z]+$'
# text = "TCGA-12-1091-01Z-00-DX2.E3A31BCE-F8DD-4B9D-AB05-A93EACFE5971.svs"

# match = re.match(pattern, text)
# if match:
#     group1 = match.group(1)
#     group2 = match.group(2)
#     print("Group 1:", group1)
#     print("Group 2:", group2)
# else:
#     print("No match found.")

# aug_func = get_srh_base_aug
# from torchvision.transforms import Compose
# from datasets.srh_dataset import OpenSRHDataset
# train_dset = OpenSRHDataset(data_root="/l/users/hasindri.watawana/hidisc/datasets/opensrh",
#                                 studies="train",
#                                 transform=Compose(aug_func()),
#                                 balance_patch_per_class=False)

# val_dset = OpenSRHDataset(data_root="/l/users/hasindri.watawana/hidisc/datasets/opensrh",
#                               studies="val",
#                               transform=Compose(aug_func()),
#                               balance_patch_per_class=False)

# print(f'train dataset length {len(train_dset)}')
# print(f'val dataset length {len(val_dset)}')

# # for idx, data in enumerate(val_dset):
# #     print(idx)

# # val_dset[13576]

#------------------------------

# import pandas as pd

# # Load the data from the text file into a DataFrame
# file_path = '/l/users/hasindri.watawana/hidisc/datasets/tcga/gdc_manifest_20230727_073249.txt'
# df = pd.read_csv(file_path, delimiter='\t')

# # Calculate the total number of unique IDs
# total_unique_ids = df['id'].nunique()

# # Calculate the total number of unique .svs filenames
# total_unique_filenames = df['filename'].nunique()

# # Calculate the total size of all data in bytes
# total_size_bytes = df['size'].sum()

# # Convert the total size to gigabytes (GB)
# total_size_gb = total_size_bytes / (1024**3)

# # Print the results
# print(f"Total unique IDs: {total_unique_ids}")
# print(f"Total unique .svs filenames: {total_unique_filenames}")
# print(f"Total size of all data (GB): {total_size_gb:.4f} GB")

#----------------------------------------

import os

# Define the directory path
directory_path = '/l/users/hasindri.watawana/hidisc/datasets/tcga'

# Initialize a variable to store the total size in bytes
total_size_bytes = 0

# Function to calculate the size of a file or directory recursively
def calculate_size(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    elif os.path.isdir(path):
        total = 0
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            total += calculate_size(item_path)
        return total
    else:
        return 0

# Traverse the directory and calculate the total size
for root, dirs, files in os.walk(directory_path):
    for file in files:
        if file.endswith('.svs'):
            file_path = os.path.join(root, file)
            total_size_bytes += calculate_size(file_path)

# Convert the total size to gigabytes (GB)
total_size_gb = total_size_bytes / (1024 ** 3)

# Print the total size in GB
print(f"Total size of .svs files in GB: {total_size_gb:.4f} GB")




