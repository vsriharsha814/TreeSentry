

# TreeSentry: Deep Learning for Deforestation Detection

## Overview

Deforestation continues to be one of the greatest threats to global climate stability, biodiversity, and ecosystem health. Traditional monitoring systems, while valuable, often miss early signs of deforestation due to their reliance on low-resolution satellite imagery and simple vegetation indices. 

**TreeSentry** is a project developed by Sri Harsha Vallabhaneni and Samhitha Sannidhi, aimed at addressing this gap by combining high-resolution satellite imagery from Planet Labs with deep learning techniques, particularly convolutional neural networks (CNNs), to detect deforestation events earlier and more accurately.

While our focus for this project is on the Amazon Rainforest — one of the most critical and vulnerable ecosystems — the methodology we develop is designed to be scalable and adaptable to forests worldwide.

## Project Motivation

- Enable early detection of small-scale deforestation events that traditional methods miss.
- Improve precision and recall of forest loss detection compared to NDVI-based baselines.
- Support conservation efforts by providing actionable and timely insights into forest health.
- Showcase a real-world application of deep learning models for environmental monitoring.

## Approach

- **Dataset:** We use the Planet Amazon Rainforest dataset (40,000+ high-resolution satellite images with multi-label annotations).
- **Model:** We fine-tune a pre-trained ResNet-50 CNN architecture to perform binary classification (deforestation vs no-deforestation).
- **Training:** The model is trained using binary cross-entropy loss, with data augmentations (flips, rotations) and class-balanced mini-batches to tackle imbalance.
- **Evaluation:** We compare our deep learning model against traditional NDVI thresholding baselines using accuracy, precision, recall, and F1 score.
- **Experiments:** In addition to quantitative metrics, we perform qualitative analysis by visually inspecting model predictions over sample areas.

## Quickstart

1. Set up a Python environment (Python 3.8+ recommended).
2. Install dependencies:
   ```
   pip install torch torchvision pandas pillow kaggle
   ```
3. Download the Planet Amazon dataset from Kaggle (after accepting competition rules):
   ```
   kaggle competitions download -c planet-understanding-the-amazon-from-space -p data/
   cd data/
   unzip train-jpg.zip
   mv train-jpg train_images
   mv train_v2.csv train_labels.csv
   ```
4. Train the model:
   ```
   python app.py
   ```
5. Make predictions on new images using the provided `predict` function.

## Team

- **Sri Harsha Vallabhaneni**
- **Samhitha Sannidhi**

Developed as part of our Neural Networks and Deep Learning course project, with a focus on solving a real-world environmental challenge using deep learning principles.

---