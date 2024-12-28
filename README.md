# Classification of Heart Sound Signals

This repository contains the implementation of a machine learning model for classifying heart sound recordings as **normal** or **abnormal**. The project leverages advanced techniques in deep learning and traditional machine learning to achieve high accuracy in heart sound classification.

## Features
- **Convolutional Neural Network (CNN):** Built and trained a CNN model to classify heart sounds with high precision.
- **Preprocessing Pipeline:** Applied noise reduction and heartbeat segmentation for clean and standardized audio inputs.
- **Model Optimization:** Fine-tuned CNN architecture by adjusting hyperparameters to enhance performance.
- **Comparative Analysis:** Used a Random Forest classifier for feature-based classification and compared results with the CNN model.

## Technologies Used
- **Deep Learning Frameworks:** TensorFlow, Keras
- **Machine Learning Libraries:** Scikit-learn
- **Audio Processing Library:** Librosa
- **Classification Model:** Random Forest

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/5upto/binary-classifier.git
    ```
2. Navigate to the project directory:
    ```bash
    cd binary-classifier
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Place your heart sound recordings in the `data` directory.
2. Preprocess the audio files using the provided scripts for noise reduction and segmentation.
3. Train the CNN model:
    ```bash
    python train_cnn.py
    ```
4. Evaluate the model:
    ```bash
    python evaluate.py
    ```
5. For comparative analysis, use the Random Forest classifier:
    ```bash
    python random_forest_classifier.py
    ```

## Results
The CNN model demonstrated superior performance in classifying heart sound recordings. Comparative analysis with the Random Forest classifier highlighted the effectiveness of deep learning for this task.

## Future Work
- Incorporate additional data augmentation techniques to enhance model robustness.
- Explore transfer learning approaches with pre-trained audio classification models.
- Develop a user-friendly interface for real-time heart sound classification.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For queries or contributions, feel free to reach out via [GitHub](https://github.com/5upto/binary-classifier).
