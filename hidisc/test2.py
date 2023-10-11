import os

# file_path = "/l/users/hasindri.watawana/hidisc/datasets/opensrh/studies_tgz/NIO_050.tgz"

# size_bytes = os.path.getsize(file_path)

# size_kb = size_bytes / 1024  # Convert bytes to kilobytes
# size_mb = size_kb / 1024  # Convert kilobytes to megabytes
# size_gb = size_mb / 1024  # Convert megabytes to gigabytes

# print(f"File Size (GB): {size_gb:.2f} GB")


import os
import tarfile
from tqdm import tqdm
import time
import os
import tarfile
from tqdm import tqdm
import subprocess
import torch

from datasets.srh_dataset import OpenSRHDataset, HiDiscDataset, process_read_im, get_srh_base_aug
from common import (setup_output_dirs, parse_args, get_exp_name,
                           config_loggers, get_optimizer_func,
                           get_scheduler_func, get_dataloaders,get_num_worker)
from datasets.improc import process_read_im, get_srh_base_aug

import warnings

#to preprocess openSRH files

# def extract_tgz_files(src_folder, dest_folder):
#     tgz_files = [f for f in os.listdir(src_folder) if f.endswith('.tgz')]
#     print(f'tgz files {len(tgz_files)}')

#     dest_folders = [f for f in os.listdir(dest_folder) if os.path.isdir(os.path.join(dest_folder, f))]
    
#     tgz_files = [tgz_file for tgz_file in tgz_files if tgz_file.replace('.tgz', '') not in dest_folders]
#     print(f'tgz files {len(tgz_files)}')

#     cant_read = []
#     for file_no, tgz_file in enumerate(tgz_files):
#         tgz_path = os.path.join(src_folder, tgz_file)
#         dest_path = dest_folder

#         # print(f'file size {(os.path.getsize(tgz_path))/(1024**3)} GB')
#         print(f'now processing file {file_no} file_name {tgz_file}')

#         try:
#             # Use subprocess.run with timeout to measure elapsed time
#             result = subprocess.run(['tar', 'tzf', tgz_path], timeout=600, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         except subprocess.TimeoutExpired:
#             print(f'cant read file {tgz_file}')
#             cant_read.append(tgz_file)
#             continue

#         # Opening the .tgz file using tarfile.open as it passed the timeout check
#         with tarfile.open(tgz_path, 'r:gz') as tar:
#             # Get the total number of members (files/directories) in the archive
#             total_members = len(tar.getmembers())
#             # Initialize tqdm to display the progress bar
#             progress_bar = tqdm(total=total_members, desc=f'Extracting {tgz_file}', unit='files')

#             # Extract each member in the archive
#             for member in tar.getmembers():
#                 tar.extract(member, path=dest_path)
#                 progress_bar.update(1)  # Update the progress bar for each extracted member
            
#             progress_bar.close()  # Close the progress bar when extraction is complete

#     print(f'Cant read files: {cant_read}')

# # Call the function with the source and destination folder paths
# src_folder = "/l/users/hasindri.watawana/hidisc/datasets/opensrh/studies_tgz"
# dest_folder = "/l/users/hasindri.watawana/hidisc/datasets/opensrh/studies2"

# extract_tgz_files(src_folder, dest_folder)

# -----------------------------------------------------------------------

import warnings


# Ignore all warnings
warnings.filterwarnings("ignore")

aug_func = get_srh_base_aug
from torchvision.transforms import Compose
from datasets.srh_dataset import OpenSRHDataset
train_dset = OpenSRHDataset(data_root="/l/users/hasindri.watawana/hidisc/datasets/opensrh",
                                studies="train",
                                transform=Compose(aug_func()),
                                balance_patch_per_class=False)

print(f'train dataset length {len(train_dset)}')

train_dset.reset_index()

train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=128,
        drop_last=False,
        pin_memory=True,
        num_workers=get_num_worker(),
        persistent_workers=True)

for batch_idx, batch_data in enumerate(train_loader):
    # The custom __getitem__ method will print the index for each iteration
    pass
