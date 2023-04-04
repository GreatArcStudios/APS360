import pandas as pd
import os
import shutil

# Read in the original CSV file
df = pd.read_csv('./data/nih_data/Data_Entry_2017.csv')

# Create a mask to separate "No Finding" rows from others
mask = df['Finding Labels'] == 'No Finding'

# Create nofind dataframe
nofind_df = df[mask][['Image Index', 'Finding Labels']]
nofind_df.columns = ['image', 'class']

# Create other dataframe
other_df = df[~mask][['Image Index', 'Finding Labels']]
other_df.columns = ['image', 'class']

# Save dataframes to CSV files
nofind_df.to_csv('nofindv2.csv', index=False)
other_df.to_csv('./csv_mappings/other_multilabel.csv', index=False)

import os
import shutil

# Set path to extracted images
extracted_images_path = './data/nih_data/'

# Set path to save the class folders and images
save_path = './data/class_folders/'

# Set number of "No Finding" images to sample
num_no_finding_samples = 1000

# Read in the other CSV file
other_df = pd.read_csv('./csv_mappings/other_multilabel.csv')

# Read in the nofind CSV file
nofind_df = pd.read_csv('./csv_mappings/nofindv2.csv')

# Sample num_no_finding_samples rows from nofind_df
nofind_sample = nofind_df.sample(n=num_no_finding_samples)

# Concatenate nofind_sample with other_df
other_df = pd.concat([other_df, nofind_sample])

# Create a dictionary to store image names for each class
class_dict = {}

# Loop through rows in other_df
for index, row in other_df.iterrows():
    # Get image name and labels for current row
    image_name = row['image']
    labels = row['class'].split('|')
    
    # Loop through labels for current image
    for label in labels:
        # Create a new key in class_dict if it doesn't already exist
        if label not in class_dict:
            class_dict[label] = []
        
        # Add image name to list of images for current label
        class_dict[label].append(image_name)

# Loop through classes in class_dict
for label, image_names in class_dict.items():
    # Create a new folder for current label if it doesn't already exist
    os.makedirs(os.path.join(save_path, label), exist_ok=True)
    
    # Loop through image names for current label
    for image_name in image_names:
        # Set source path for current image (searching all subfolders under extracted_images_path)
        src_path = None
        
        for root, dirs, files in os.walk(extracted_images_path):
            if src_path is not None:
                break
            
            if image_name in files:
                src_path = os.path.join(root, image_name)
                break
        
        if src_path is None:
            print(f'Warning: Could not find {image_name} under {extracted_images_path}')
            continue
        
        # Set destination path for current image (inside the folder created earlier for current label)
        dst_path = os.path.join(save_path, label, image_name)
        
        # Copy the file from src_path to dst_path (use shutil.move instead of shutil.copy if you want to move instead of copy)
        shutil.copy(src_path, dst_path)