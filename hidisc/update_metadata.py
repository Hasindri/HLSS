import os
import json
from tqdm import tqdm

data_folder = "/l/users/hasindri.watawana/hidisc/datasets/opensrh/studies"
opensrh_folder = "/l/users/hasindri.watawana/hidisc/datasets/opensrh"
# input_json_file = "/l/users/hasindri.watawana/hidisc/datasets/opensrh/meta/opensrh.json"
# output_json_file = "/l/users/hasindri.watawana/hidisc/datasets/opensrh/meta/updated2_opensrh.json"

# # Function to check if a directory exists in the data folder
# def directory_exists(directory_path):
#     return os.path.exists(os.path.join(data_folder, directory_path))

# # Function to check if a file exists in the data folder
# def file_exists(file_path):
#     return os.path.exists(os.path.join(opensrh_folder, file_path))

# # Read the contents of the opensrh.json file and load it as a dictionary
# with open(input_json_file, "r") as f:
#     metadata = json.load(f)

# # Update the metadata dictionary to remove patient and slide data that don't exist
# patients_to_remove = []
# for patient_id, patient_data in tqdm(metadata.items()):
#     if not directory_exists(patient_id):
#         # Patient folder does not exist, mark for removal
#         patients_to_remove.append(patient_id)
#     else:
#         slides_to_remove = []
#         for slide_id, slide_data in patient_data["slides"].items():
#             slide_folder_path = os.path.join(patient_id, slide_id)
#             if not directory_exists(slide_folder_path):
#                 # Slide folder does not exist, mark for removal
#                 slides_to_remove.append(slide_id)
#             else:
#                 # Check patch paths and remove if patch file doesn't exist
#                 tumor_patches = [patch for patch in slide_data["tumor_patches"] if file_exists(patch)]
#                 normal_patches = [patch for patch in slide_data["normal_patches"] if file_exists(patch)]
#                 nondiagnostic_patches = [patch for patch in slide_data["nondiagnostic_patches"] if file_exists(patch)]
#                 # print(f'tumor patches of slide {slide_id} are {len(tumor_patches)}')
#                 # print(f'normal patches of slide {slide_id} are {len(normal_patches)}')
#                 # print(f'non patches of slide {slide_id} are {len(nondiagnostic_patches)}')
#                 slide_data["tumor_patches"] = tumor_patches
#                 slide_data["normal_patches"] = normal_patches
#                 slide_data["nondiagnostic_patches"] = nondiagnostic_patches
            
        
#         # Remove slides marked for removal
#         for slide_id in slides_to_remove:
#             del patient_data["slides"][slide_id]

# # Remove patients marked for removal
# print(f'patients to remove {patients_to_remove}')
# for patient_id in patients_to_remove:
#     del metadata[patient_id]

# # Save the updated metadata as a new JSON file
# with open(output_json_file, "w") as f:
#     json.dump(metadata, f, indent=4)

# print("Updated metadata has been saved to:", output_json_file)

# --------------------------------------------------------

# import json

# updated_json = "/l/users/hasindri.watawana/hidisc/datasets/opensrh/meta/updated2_opensrh.json"
# old_split = "/l/users/hasindri.watawana/hidisc/datasets/opensrh/meta/train_val_split.json"
# updated_split = "/l/users/hasindri.watawana/hidisc/datasets/opensrh/meta/updated2_train_val_split.json"


# # Load the updated metadata from the previous step
# with open(updated_json, "r") as f:
#     updated_metadata = json.load(f)

# # Load the original train_val_split.json
# with open(old_split, "r") as f:
#     train_val_split = json.load(f)

# # Remove patient IDs that don't exist in the updated metadata
# train_val_split["train"] = [patient_id for patient_id in train_val_split["train"] if patient_id in updated_metadata]
# train_val_split["val"] = [patient_id for patient_id in train_val_split["val"] if patient_id in updated_metadata]

# # Save the updated train_val_split.json as a new JSON file
# with open(updated_split, "w") as f:
#     json.dump(train_val_split, f, indent=4)

# print("Updated train_val_split.json has been saved to: updated_train_val_split.json")

# ------------------------------------

# import json

# # Paths to the JSON files
# updated_json = "/l/users/hasindri.watawana/hidisc/datasets/opensrh/meta/updated_opensrh.json"
# updated_split = "/l/users/hasindri.watawana/hidisc/datasets/opensrh/meta/updated_train_val_split.json"

# # Load the JSON data from the files
# with open(updated_json, 'r') as json_file:
#     metadata = json.load(json_file)

# with open(updated_split, 'r') as json_file:
#     split_data = json.load(json_file)

# # Get the patient IDs from the train and val splits
# train_patient_ids = split_data["train"]
# val_patient_ids = split_data["val"]

# # Check if all patient IDs in train split are in metadata
# train_ids_in_metadata = all(patient_id in metadata for patient_id in train_patient_ids)

# # Check if all patient IDs in val split are in metadata
# val_ids_in_metadata = all(patient_id in metadata for patient_id in val_patient_ids)

# # Print the results
# if train_ids_in_metadata:
#     print("All patient IDs in the train split are present in the metadata.")
# else:
#     print("Some patient IDs in the train split are not present in the metadata.")

# if val_ids_in_metadata:
#     print("All patient IDs in the val split are present in the metadata.")
# else:
#     print("Some patient IDs in the val split are not present in the metadata.")

# ------------------------------------

import json

# Paths to the JSON files
updated_json = "/l/users/hasindri.watawana/hidisc/datasets/opensrh/meta/updated2_opensrh.json"
updated_split = "/l/users/hasindri.watawana/hidisc/datasets/opensrh/meta/updated2_train_val_split.json"

# Load the JSON data from the files
with open(updated_json, 'r') as json_file:
    metadata = json.load(json_file)

with open(updated_split, 'r') as json_file:
    split_data = json.load(json_file)

# Get the patient IDs from the train and val splits
train_patient_ids = set(split_data["train"])
val_patient_ids = set(split_data["val"])

# Get the patient IDs from the metadata
metadata_patient_ids = set(metadata.keys())

# Check if patient IDs in train + val split = patient IDs in metadata
if metadata_patient_ids == train_patient_ids.union(val_patient_ids):
    print("Patient IDs in the train + val split are equal to patient IDs in metadata.")
else:
    print("Patient IDs in the train + val split are not equal to patient IDs in metadata.")

# Check if metadata doesn't have any extra patient IDs not in splits
if metadata_patient_ids.issubset(train_patient_ids.union(val_patient_ids)):
    print("Metadata doesn't have any extra patient IDs not present in the splits.")
else:
    print("Metadata has extra patient IDs not present in the splits.")

    # ------------------------
