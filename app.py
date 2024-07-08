from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import numpy as np
import h5py
import time
from skimage import io
from skimage.color import rgba2rgb
from skimage.transform import resize
from scipy import spatial
from models.basic.feature_extraction import extract_custom_features
from models.VGG16.VGG_feature_extractor import VGGNet
from models.VGG19.VGG19_feature_extractor import FeatureExtractorVGG19
from models.xception.Xception_feature_extractor import FeatureExtractorXception
from models.inceptionv3.inception_feature_extractor import FeatureExtractorInception

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load features and labels for custom method
h5f = h5py.File('data/CustomFeatures.h5', 'r')
feats = h5f['dataset_1'][:]
labels = h5f['dataset_2'][:]
h5f.close()

# Load features and labels for VGG16 method
h5f_vgg = h5py.File('data/VGG16Features.h5', 'r')
feats_vgg = h5f_vgg['dataset_1'][:]
labels_vgg = h5f_vgg['dataset_2'][:]
h5f_vgg.close()

# Load features and labels for VGG19 method
h5f_vgg19 = h5py.File('data/VGG19Features.h5', 'r')
feats_vgg19 = h5f_vgg19['dataset_1'][:]
labels_vgg19 = h5f_vgg19['dataset_2'][:]
h5f_vgg19.close()

# Load features and labels for Xception method
h5f_xception = h5py.File('data/XceptionFeatures.h5', 'r')
feats_xception = h5f_xception['dataset_1'][:]
labels_xception = h5f_xception['dataset_2'][:]
h5f_xception.close()

# Load features and labels for Inception method
h5f_inception = h5py.File('data/InceptionFeatures.h5', 'r')
feats_inception = h5f_inception['dataset_1'][:]
labels_inception = h5f_inception['dataset_2'][:]
h5f_inception.close()

# Define route for index page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        method = request.form.get('method')
        if method == 'custom':
            return jsonify(get_similar_images(filepath, method='custom'))
        elif method == 'vgg16':
            return jsonify(get_similar_images(filepath, method='vgg16'))
        elif method == 'vgg19':
            return jsonify(get_similar_images(filepath, method='vgg19'))
        elif method == 'xception':
            return jsonify(get_similar_images(filepath, method='xception'))
        elif method == 'inception':
            return jsonify(get_similar_images(filepath, method='inception'))
        else:
            return jsonify({"error": "Invalid method"})

def calculate_similarity(query_features, database_features):
    scores = []
    for i in range(database_features.shape[0]):
        score = 1 - spatial.distance.cosine(query_features, database_features[i])
        scores.append(score)
    scores = np.array(scores)
    return scores

def get_similar_images(image_path, method='custom'):
    img = io.imread(image_path)
    if len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)
    if img.shape[-1] == 4:
        img = rgba2rgb(img)
    img = resize(img, (256, 256))

    start_time = time.time()  # Start measuring time

    if method == 'custom':
        query_features = extract_custom_features(img)
        scores = calculate_similarity(query_features, feats)
        rank_ID = np.argsort(scores)[::-1]
        top_images = [(labels[rank_ID[i]].decode('utf-8'), scores[rank_ID[i]]) for i in range(5)]

    elif method == 'vgg16':
        model = VGGNet()
        query_features = model.extract_feat(image_path)
        scores = calculate_similarity(query_features, feats_vgg)
        rank_ID = np.argsort(scores)[::-1]
        top_images = [(labels_vgg[rank_ID[i]].decode('utf-8'), scores[rank_ID[i]]) for i in range(5)]

    elif method == 'vgg19':
        model = FeatureExtractorVGG19()
        query_features = model.extract(image_path)
        scores = calculate_similarity(query_features, feats_vgg19)
        rank_ID = np.argsort(scores)[::-1]
        top_images = [(labels_vgg19[rank_ID[i]].decode('utf-8'), scores[rank_ID[i]]) for i in range(5)]

    elif method == 'xception':
        model = FeatureExtractorXception()
        query_features = model.extract(image_path)
        scores = calculate_similarity(query_features, feats_xception)
        rank_ID = np.argsort(scores)[::-1]
        top_images = [(labels_xception[rank_ID[i]].decode('utf-8'), scores[rank_ID[i]]) for i in range(5)]

    elif method == 'inception':
        model = FeatureExtractorInception()
        query_features = model.extract(image_path)
        scores = calculate_similarity(query_features, feats_inception)
        rank_ID = np.argsort(scores)[::-1]
        top_images = [(labels_inception[rank_ID[i]].decode('utf-8'), scores[rank_ID[i]]) for i in range(5)]

    end_time = time.time()  # End measuring time
    extraction_time = end_time - start_time  # Calculate the elapsed time

    return {"top_images": top_images, "extraction_time": extraction_time}

if __name__ == '__main__':
    app.run(debug=True)
