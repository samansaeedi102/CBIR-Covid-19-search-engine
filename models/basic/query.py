import numpy as np
import h5py
import time
from skimage import io
from skimage.color import rgba2rgb
from skimage.transform import resize
from scipy import spatial
from feature_extraction import extract_custom_features

# Load features and image paths
h5f = h5py.File('data/CustomFeatures.h5', 'r')
feats = h5f['dataset_1'][:]
image_paths = h5f['dataset_2'][:]
h5f.close()

def find_similar_images(image_path, top_n=5):
    img = io.imread(image_path)

    # Handle grayscale images
    if len(img.shape) == 2:
        img = np.stack((img,)*3, axis=-1)

    # Handle RGBA images
    if img.shape[-1] == 4:
        img = rgba2rgb(img)
    
    img = resize(img, (256, 256))

    # Measure feature extraction time
    start_time = time.time()
    query_features = extract_custom_features(img)
    extraction_time = time.time() - start_time
    
    scores = []
    for i in range(feats.shape[0]):
        score = 1 - spatial.distance.cosine(query_features, feats[i])
        scores.append(score)
    
    scores = np.array(scores)
    rank_ID = np.argsort(scores)[::-1]
    
    top_image_paths = [image_paths[idx].decode('utf-8') for idx in rank_ID[:top_n]]
    
    return top_image_paths, extraction_time
