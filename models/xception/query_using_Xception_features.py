import numpy as np
import h5py
from scipy import spatial
from Xception_feature_extractor import XceptionNet

# Load features and labels
h5f = h5py.File('data/XceptionFeatures.h5', 'r')
feats = h5f['dataset_1'][:]
imgNames = h5f['dataset_2'][:]
h5f.close()

# Read the query image
queryImg = "query_images/query_image.jpg"

print("Searching for similar images.")

# Initialize XceptionNet model
model = XceptionNet()

# Extract Features
X = model.extract_feat(queryImg)

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
print("Top %d images in order are: " % maxres, imlist)
