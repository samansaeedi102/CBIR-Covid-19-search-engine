import numpy as np
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from numpy.linalg import norm
from PIL import Image 

class FeatureExtractorVGG19:
    def __init__(self):
        self.input_shape = (224, 224, 3)
        self.weight = 'imagenet'
        self.pooling = 'max'
        self.model = VGG19(weights=self.weight, input_shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2]), pooling=self.pooling, include_top=False)
        self.model.predict(np.zeros((1, 224, 224, 3)))

    def extract(self, img_path):
        # Open the image file
        img = Image.open(img_path)
        # Resize the image and convert it to RGB format
        img = img.resize((224, 224)).convert("RGB")
        # Convert the image to an array and preprocess it
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        # Extract features using the VGG19 model
        features = self.model.predict(img_array)
        # Normalize the feature vector
        norm_features = features[0] / norm(features[0])
        return norm_features
