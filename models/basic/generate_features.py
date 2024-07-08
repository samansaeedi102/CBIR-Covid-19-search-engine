import os
import h5py
import numpy as np
from skimage import io
from feature_extraction import extract_custom_features 

# Define paths
path_covid = "E:\information retrieval\CBIR\images\COVID"  
path_non_covid = "E:/information retrieval/CBIR/images/Non-COVID"
output_file = 'E:\information retrieval\CBIR\data\CustomFeatures.h5'

# Create lists to hold the features and image paths
feats = []
image_paths = []

# Process COVID images
for im in os.listdir(path_covid):
    print("Extracting features from COVID image - ", im)
    img = io.imread(os.path.join(path_covid, im))
    if len(img.shape) == 2:  # If grayscale
        img = np.stack((img,) * 3, axis=-1)
    elif img.shape[2] == 4:  # If RGBA
        img = img[:, :, :3]
    X = extract_custom_features(img)
    feats.append(X)
    image_paths.append(os.path.join(path_covid, im))

# Process non-COVID images
for im in os.listdir(path_non_covid):
    print("Extracting features from non-COVID image - ", im)
    img = io.imread(os.path.join(path_non_covid, im))
    if len(img.shape) == 2:  # If grayscale
        img = np.stack((img,) * 3, axis=-1)
    elif img.shape[2] == 4:  # If RGBA
        img = img[:, :, :3]
    X = extract_custom_features(img)
    feats.append(X)
    image_paths.append(os.path.join(path_non_covid, im))

# Convert lists to numpy arrays
feats = np.array(feats)
image_paths = np.array(image_paths, dtype='S')  # Use dtype='S' for string data

# Write the features and image paths to an HDF5 file
h5f = h5py.File(output_file, 'w')
h5f.create_dataset('dataset_1', data=feats)
h5f.create_dataset('dataset_2', data=image_paths)
h5f.close()
