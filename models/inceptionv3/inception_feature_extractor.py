import numpy as np
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from numpy import linalg as LA

class FeatureExtractorInception:
    def __init__(self):
        self.model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

    def extract(self, img_path):
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        feat = self.model.predict(x)
        norm_feat = feat[0] / LA.norm(feat[0])

        return norm_feat
