from VGG19_feature_extractor import FeatureExtractorVGG19
import numpy as np
import h5py
import matplotlib.pyplot as plt
from PIL import Image
from scipy import spatial

# Load the features database (HDF5 file)
h5f = h5py.File("E:/information retrieval/CBIR/data/VGG19Features.h5", 'r')
feats = h5f['dataset_1'][:]
imgNames = h5f['dataset_2'][:]
h5f.close()

# Read the query image
queryImg = "E:/information retrieval/CBIR/query_images/monkey4.jpg"

print("Searching for similar images")

# Initialize VGG19 model
model = FeatureExtractorVGG19()

# Extract Features
img = Image.open(queryImg)
X = model.extract(img)

# Compute the Cosine distance between 1-D arrays
scores = []
for i in range(feats.shape[0]):
    score = 1 - spatial.distance.cosine(X, feats[i])
    scores.append(score)
scores = np.array(scores)
rank_ID = np.argsort(scores)[::-1]
rank_score = scores[rank_ID]

maxres = 5
imlist = [imgNames[index].decode('utf-8') for i, index in enumerate(rank_ID[0:maxres])]
print("top %d images in order are: " % maxres, imlist)
