import os
import openslide
from PIL import Image
from tqdm import tqdm
import re


def extract_patches(svs_file, patient_id, slide_id, uuid_folder):
    slide = openslide.OpenSlide(svs_file)
    slide_width, slide_height = slide.dimensions

    print(f'slide_width {slide_width}')
    print(f'slide_height {slide_height}')
    patch_size = 300
    patch_id = 0

    for x in tqdm(range(0, slide_width - patch_size + 1, patch_size)):
        for y in range(0, slide_height - patch_size + 1, patch_size):
            patch = slide.read_region((x, y), 0, (patch_size, patch_size))

            # Check if the patch contains non-white pixels (remove borders)
            if patch.getextrema() != ((255, 255),):
                # Save the patch only if it is entirely within the slide boundaries
                patch_folder = os.path.join(uuid_folder, "patches")
                patch_id += 1
                patch_path = os.path.join(patch_folder, f"{patient_id}-{slide_id}-{patch_id}.tif")
                patch.save(patch_path)

    slide.close()


def main():
    root_dir = "/l/users/hasindri.watawana/hidisc/datasets/tcga"
    # processed_folders = []  # Keep track of already processed patient folders

    # Check if 'tcga_glioma' directory exists and create if it doesn't
    # glioma_dir = "/l/users/hasindri.watawana/hidisc/datasets/tcga_glioma/studies"
    # os.makedirs(glioma_dir, exist_ok=True)
    # processed_folders = os.listdir(glioma_dir)

    # Update 'root_dir' to tcga_glioma/studies
    # root_dir = glioma_dir

    for uuid_folder in os.listdir(root_dir):
        print(f'uuid folder {uuid_folder}')
        uuid_folder_path = os.path.join(root_dir, uuid_folder)

        patch_folder = os.path.join(uuid_folder_path,"patches")
        print(f'patch folder {patch_folder}')

        if not os.path.exists(patch_folder) or not os.listdir(patch_folder):

            for svs_file in os.listdir(uuid_folder_path):
                if svs_file.endswith('.svs'):
                    pattern = r'^(TCGA-\d{2}-\d{4})-\d{2}[A-Z]-\d{2}-DX(\d)\.[A-F0-9-]+\.[a-z]+$'

                    match = re.match(pattern, svs_file)
                    if match:
                        patient_id = match.group(1)
                        slide_id = match.group(2)
                        print("patient_id:", patient_id)
                        print("slide_id:", slide_id)
                    else:
                        print("No match found.")

                    os.makedirs(patch_folder, exist_ok=True)

                    # Extract patches and save in 'patches' folder
                    extract_patches(os.path.join(uuid_folder_path, svs_file), patient_id, slide_id, uuid_folder)

        break

if __name__ == "__main__":
    main()
