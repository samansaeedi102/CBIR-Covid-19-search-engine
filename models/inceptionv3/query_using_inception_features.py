import numpy as np
import h5py
from scipy import spatial
from inception_feature_extractor import FeatureExtractorInception

# Load features and labels
h5f = h5py.File('data/InceptionFeatures.h5', 'r')
feats = h5f['dataset_1'][:]
imgNames = h5f['dataset_2'][:]
h5f.close()

# Read the query image
queryImg = "query_images/query_image.jpg"

print("Searching for similar images.")

# Initialize FeatureExtractorInception model
model = FeatureExtractorInception()

# Extract Features
X = model.extract(queryImg)

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
