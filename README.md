# Bank Classification Project

This project focuses on classifying images of banknotes using deep learning models. The dataset consists of images from five different classes: Aomsin, Bangkok, KasikornThai, KrungThai, and SCB. The project involves several steps, including data preprocessing, model training, and evaluation.

## Dataset Overview
- **Training Data**: 1250 images (250 images per class)
- **Testing Data**: 50 images (10 images per class)

## Models Used
- **MobileNetV2**: A lightweight and efficient model suitable for mobile and embedded vision applications.
- **ResNet50V2**: A deeper convolutional neural network known for its performance in image classification tasks.
- **DenseNet121**: A model that connects each layer to every other layer in a feed-forward fashion, improving feature propagation.
- **EfficientNetV2B0**: A scalable and efficient model that balances accuracy and computational efficiency.

## Training Process
- **Cross-Validation**: The training process uses K-Fold cross-validation to ensure robust model evaluation.
- **Early Stopping**: Implemented to prevent overfitting by stopping training when validation accuracy does not improve.
- **Model Checkpoints**: Best models are saved during training based on validation accuracy.

## Evaluation Metrics
- **Accuracy**: Measures the proportion of correctly classified images.
- **Precision**: Indicates the accuracy of positive predictions.
- **Recall**: Measures the ability of the model to find all positive samples.
- **F1-Score**: Harmonic mean of precision and recall, providing a balance between the two.

## Results
- The best model achieved high accuracy and F1-score, indicating strong performance in classifying banknote images.
- Detailed results, including precision, recall, and F1-scores, are saved in CSV files for each fold and summarized in a final report.

## Usage
To replicate the results, follow these steps:
1. **Data Preparation**: Organize the dataset into training and testing folders.
2. **Model Training**: Run the training script with the desired model (MobileNetV2, ResNet50V2, DenseNet121, or EfficientNetV2B0).
3. **Evaluation**: Use the provided scripts to evaluate the model on the test dataset and generate performance metrics.

## Dependencies
- TensorFlow
- Keras
- Scikit-learn
- Matplotlib
- Pandas
- Numpy

## Conclusion
This project demonstrates the effectiveness of various deep learning models in classifying banknote images. The results show that these models can achieve high accuracy and reliability, making them suitable for real-world applications in banking and finance.
