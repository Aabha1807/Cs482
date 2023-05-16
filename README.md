# Classifying Facial Emotions via Machine Learning

This repository contains the code and resources for a facial emotion classification project using machine learning techniques. The project aims to accurately classify human emotions based on facial expressions in images or real-time video streams.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Models](#models)
- [Model Training](#model-training)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Classifying facial emotions is an important task in the field of computer vision and human-computer interaction. Understanding human emotions can have numerous applications, such as in improving human-robot interaction, virtual reality experiences, and mental health monitoring.

This project utilizes machine learning techniques, specifically deep learning, to build a model that can accurately classify different facial emotions. The model is trained on a labeled dataset of facial images, and once trained, it can predict the emotions present in new, unseen images.

The repository provides the necessary code, documentation, and resources to reproduce the experiments and train your own facial emotion classification model.

## Installation

To run this project on your local machine, please follow these steps:

1. Clone the repository:

   ```
   git clone https://github.com/your-username/facial-emotion-classification.git
   ```

2. Install the required dependencies. It is recommended to use a virtual environment:

   ```
   cd facial-emotion-classification
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Download the pre-trained model weights, if provided, and place them in the appropriate directory.

4. You're ready to go! Now you can explore the code and run the project.

## Dataset

The facial emotion classification model in this project utilizes the [Face Expression Recognition Dataset](https://www.kaggle.com/jonathanoheix/face-expression-recognition-dataset) available on Kaggle. The dataset consists of facial images labeled with seven emotion classes: angry, disgust, fear, happy, neutral, sad, and surprise.

To use the dataset for training the model, follow these steps:

1. Download the dataset from the Kaggle website.
2. Extract the dataset files to a directory of your choice.
3. The dataset is organized into separate folders for each emotion class.
4. We have only used 4 emotions in the following code. Transform you dataset accordingly.
5. Split the dataset into training, validation, and testing sets according to your requirements. It is recommended to have a balanced distribution of samples across all emotion classes in each set.
6. Update the dataset path and file structure information in the project code or configuration files, if necessary.

Ensure that you comply with the dataset's license and usage restrictions outlined by the original author on the Kaggle page.

 or 

You can direct find the datset in the repository. 

## Models

In this project, we have implemented three types of models for facial emotion classification:

1. **CNN (Convolutional Neural Network)**:
   - This model utilizes convolutional layers to extract meaningful features from facial images.
   - It consists of multiple convolutional layers, followed by pooling layers to downsample the spatial dimensions.
   - The extracted features are then passed through fully connected layers for classification.
   - The CNN model has shown excellent performance in capturing spatial relationships in facial expressions.

2. **Bag of Words**:
   - This model represents facial images using a bag-of-words approach commonly used in natural language processing.
   - The facial images are first preprocessed to extract local features, such as keypoints or patches.
   - Next, these local features are quantized into visual words using clustering algorithms like k-means.
   - Finally, the image is represented by a histogram of visual word occurrences, which serves as input to a classifier.
   - The Bag of Words model offers a simple yet effective way to represent facial expressions based on local visual features.

3. **Bag of Words with Means from Scratch**:
   - This model extends the Bag of Words approach by considering not only visual word occurrences but also the spatial distribution of these words.
   - In addition to quantizing local features into visual words, the model calculates the mean location of each visual word within the image.
   - By including spatial information, the model captures the spatial layout of facial expression cues.
   - The Bag of Words with Means from Scratch model provides a more robust representation by incorporating both visual word occurrences and their spatial information.

Each model has its own strengths and weaknesses, and their performance may vary depending on the dataset and task requirements. Experimenting with different models can help identify the most suitable approach for facial emotion classification in specific scenarios.

## Model Training

To train a facial emotion classification model, follow these steps:

1. Prepare your dataset by organizing the images into appropriate directories according to their emotion labels.

2. Configure the training parameters such as batch size, learning rate, and network architecture in the `config.py` file.

3. Run the training script:

   ```
   python train.py
   ```

   This script will load the dataset, create the model architecture, and train the model using the specified parameters. The trained model will be saved in the `models/` directory.

4. Monitor the training process by inspecting the loss and accuracy metrics. Adjust the training parameters if needed.

5. Evaluate the trained model on the validation set to assess its performance and make any necessary adjustments to improve it.

## Usage

Once you have a trained model, you can use it to classify facial emotions on new images or real-time video streams.

1. To classify emotions on a single image, use the `classify_image.py` script:

   ```
   python classify_image.py --image path/to/image.jpg
   ```

   The script will load the specified image, preprocess it, and predict the emotion using the trained model.

2. For real-time emotion classification on a video stream, run the `classify_video.py` script:

   ```
   python classify_video.py
   ```

   The script will access your webcam and classify the emotions in the video stream.

## Contributing

Contributions to this project are welcome. You can contribute by adding new features, improving the existing code, or fixing issues. Please refer to the [contribution guidelines](CONTRIBUTING.md) for more information.

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use and modify the code according to your needs.
