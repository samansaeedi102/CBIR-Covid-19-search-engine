from VGG_feature_extractor import VGGNet

import numpy as np
import h5py

import matplotlib.pyplot as plt

# read features database (h5 file)
h5f = h5py.File("VGG16Features.h5",'r')
feats = h5f['dataset_1'][:]
imgNames = h5f['dataset_2'][:]
h5f.close()
        
  
#Read the query image
queryImg = "query_images/monkey4.jpg"

print(" searching for similar images")

# init VGGNet16 model
model = VGGNet()

# #Extract Features
X = model.extract_feat(queryImg)


# Compute the Cosine distance between 1-D arrays
scores = []
from scipy import spatial
for i in range(feats.shape[0]):
    score = 1-spatial.distance.cosine(X, feats[i])
    scores.append(score)
scores = np.array(scores)   
rank_ID = np.argsort(scores)[::-1]
rank_score = scores[rank_ID]

maxres = 5
imlist = [imgNames[index] for i,index in enumerate(rank_ID[0:maxres])]
print("top %d images in order are: " %maxres, imlist)


