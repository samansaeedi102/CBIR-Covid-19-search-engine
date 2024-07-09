# COVID-19 Search Engine

## Project Description

The COVID-19 Search Engine is a Content-Based Image Retrieval (CBIR) system designed to assist in the diagnostic process by retrieving similar lung CT scan images from a dataset. This system uses deep learning models (Inception-V3, VGG-16, VGG-19, Xception) and a custom feature extraction method to compare and retrieve images based on visual features. This tool aims to streamline the retrieval and analysis of COVID-19-related images, aiding healthcare professionals in making informed clinical decisions swiftly.

## Features

- Implementation of deep learning models (Inception-V3, VGG-16, VGG-19, Xception)
- Custom feature extraction method for specialized image analysis
- Comprehensive dataset of COVID-19 and non-COVID-19 CT scan images
- User-friendly web application for image upload and retrieval
- Evaluation and validation of the CBIR system using standard metrics

## Setup Instructions

### Prerequisites

- Python 3.x
- Virtual Environment (`venv`)

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/COVID-19-Search-Engine.git
    cd COVID-19-Search-Engine
    ```

2. Create a virtual environment:
    ```sh
    python -m venv venv
    ```

3. Activate the virtual environment:

    - On Windows:
      ```sh
      venv\Scripts\activate
      ```
    - On macOS and Linux:
      ```sh
      source venv/bin/activate
      ```

4. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Running the Feature Extractors

To run the feature extractor for each model and save the `.h5` files in the `data` folder, follow these steps:

1. Navigate to the directory of each model:
    ```sh
    cd models/inceptionv3
    ```
    ```sh
    cd models/vgg16
    ```
    ```sh
    cd models/vgg19
    ```
    ```sh
    cd models/xception
    ```
    cd models/basic
    ```

2. Run the feature extraction script for each model:
    ```sh
    python *feature_extractor*.py
    ```

### Running the Web Application

1. Run the web application:
    ```sh
    python app.py
    ```

2. Open your web browser and go to:
    ```sh
    http://127.0.0.1:5000
    ```

## Usage

1. Upload a CT scan image using the web interface and choose the method.
2. The CBIR system retrieves and displays similar images from the dataset based on visual features.
3. Review the retrieved images to assist in diagnostic processes.

## Evaluation and Metrics

The CBIR system is evaluated using precision, recall, and F1-score to ensure its effectiveness and reliability in real-world scenarios.

## Acknowledgements

- This project utilizes pre-trained models and libraries such as TensorFlow, Keras, and scikit-learn.
