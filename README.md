# CIFAR-10 Image Classification

-----

This repository contains the Jupyter Notebook `cifar-10.ipynb` which implements an image classification solution for the **CIFAR-10** dataset. The goal of this project is to accurately classify images into one of 10 categories.

## Competition Details

  * **Kaggle Competition Link:** [CIFAR-10 Competition](https://www.kaggle.com/competitions/cifar-10/submissions)
  * **Evaluation Metric:** Accuracy
  * **Achieved Score:** 0.91070

## Project Overview

The solution utilizes a pre-trained **ResNet18** model from `torchvision.models` and fine-tunes it for the CIFAR-10 classification task. The core steps involved are:

1.  **Data Extraction:** Unpacking the `train.7z` and `test.7z` archives containing the CIFAR-10 images.
2.  **Data Preparation:**
      * Loading and preprocessing `trainLabels.csv`.
      * Using `LabelEncoder` to convert categorical labels into numerical format.
      * Creating a custom PyTorch `Dataset` (`CustomImageDataset`) to handle image loading and label mapping for the training data.
      * Creating another custom PyTorch `Dataset` (`ComfortableData`) for the test data.
3.  **Data Augmentation and Transformation:** Applying transformations such as resizing, converting to tensor, and normalizing images using `transforms.Compose` from `torchvision.transforms`.
4.  **Model Definition:**
      * Defining a `ResNetClassifier` class based on `resnet18`.
      * Freezing earlier layers of the pre-trained ResNet18 and unfreezing only `layer4` and the final `fc` (fully connected) layer to enable fine-tuning.
      * Modifying the final fully connected layer to output 10 classes, matching the CIFAR-10 dataset.
5.  **Training:**
      * Using `CrossEntropyLoss` as the criterion and `Adam` optimizer.
      * Training the model for 2 epochs on the prepared training data.
      * Monitoring loss and accuracy during training.
6.  **Inference and Submission:**
      * Making predictions on the test dataset.
      * Converting the predicted numerical labels back to their original string labels using `le.inverse_transform`.
      * Generating a `submission.csv` file in the required Kaggle format.

## Setup and Running the Notebook

To run this notebook, you'll need a Kaggle environment or a local setup with the necessary libraries.

### Prerequisites

  * Python 3.x
  * `py7zr`
  * `pandas`
  * `numpy`
  * `scikit-learn`
  * `Pillow`
  * `torch`
  * `torchvision`

### Installation

You can install the required Python packages using pip:

```bash
pip install py7zr pandas numpy scikit-learn Pillow torch torchvision
```

### Running the Notebook

1.  **Download the data:** Download the `cifar-10.zip` dataset from the Kaggle competition page and extract its contents into a directory accessible by your notebook (e.g., `/kaggle/input/cifar-10/` if on Kaggle).
2.  **Open the notebook:** Open `cifar-10.ipynb` in a Jupyter environment (Jupyter Lab, Jupyter Notebook, Google Colab, or Kaggle Notebooks).
3.  **Run all cells:** Execute all cells in the notebook sequentially. The script will perform data extraction, model training, and generate the `submission.csv` file.

