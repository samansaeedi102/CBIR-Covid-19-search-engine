import os
import h5py
import numpy as np
from PIL import Image 
from VGG19_feature_extractor import FeatureExtractorVGG19

# Define paths to the COVID and non-COVID image folders
covid_path = "E:/information retrieval/CBIR/images/COVID/"
non_covid_path = "E:/information retrieval/CBIR/images/non-COVID/"

# Collect all image paths from both folders
covid_img_list = [os.path.join(covid_path, f) for f in os.listdir(covid_path)]
non_covid_img_list = [os.path.join(non_covid_path, f) for f in os.listdir(non_covid_path)]

# Combine the lists of image paths
img_list = covid_img_list + non_covid_img_list

print("Start feature extraction.")

# Initialize VGG19 model
model = FeatureExtractorVGG19()

feats = []
names = []

# Iterate through all images to extract features
for im_path in img_list:
    print("Extracting features from image - ", im_path)
    img = Image.open(im_path) 
    X = model.extract(im_path)

    feats.append(X)
    names.append(os.path.basename(im_path)) 

feats = np.array(feats)

# Define the output HDF5 file path
output_file = "E:/information retrieval/CBIR/data/VGG19Features.h5"

print("Writing feature extraction results to HDF5 file.")

# Create HDF5 file and store features and corresponding image names
with h5py.File(output_file, 'w') as h5f:
    h5f.create_dataset('dataset_1', data=feats)
    h5f.create_dataset('dataset_2', data=np.string_(names))

print("Feature extraction completed and saved to", output_file)
