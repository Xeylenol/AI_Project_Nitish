# 🔢 Handwritten Digit Recognition Project

Machine learning project that recognizes handwritten digits (0-9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset. The project includes both model training and an interactive GUI application.

## 📋 Project Overview

- **Dataset**: MNIST (70,000 grayscale images of handwritten digits)
- **Model**: Convolutional Neural Network (CNN)
- **Framework**: TensorFlow/Keras
- **GUI**: Tkinter-based drawing interface
- **Expected Accuracy**: ~99%

## 🎯 Features

- ✅ Complete CNN model training pipeline
- ✅ Data visualization and analysis
- ✅ Model evaluation with confusion matrix
- ✅ Interactive GUI for drawing digits
- ✅ Real-time digit recognition
- ✅ Confidence scores and probability distribution
- ✅ Easy-to-use interface

## 📁 Project Structure

```
AI_Project_Nitish/
│
├── mnist_digit_recognition.ipynb   # Jupyter notebook for model training
├── digit_recognizer_gui.py         # GUI application for digit recognition
├── README.md                        # Project documentation
├── requirements.txt                 # Python dependencies
│
└── models/                          # Directory for saved models
    ├── mnist_cnn_model.keras        # Trained model (Keras format)
    └── mnist_cnn_model.h5           # Trained model (H5 format)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- Virtual environment (recommended)

### Installation

1. **Clone or navigate to the project directory**
   ```powershell
   git clone https://github.com/Xeylenol/AI_Project_Nitish.git
   cd AI_Project_Nitish
   ```

2. **Activate your virtual environment** (if using one)
   ```powershell
   .venv\Scripts\Activate.ps1
   ```

3. **Install required packages**
   ```powershell
   pip install numpy matplotlib jupyter torch tensorflow pillow opencv-python scikit-learn seaborn
   ```

## 📊 Training the Model

1. **Open the Jupyter notebook**
   ```powershell
   jupyter notebook mnist_digit_recognition.ipynb
   ```

2. **Run all cells in the notebook**
   - The notebook will:
     - Load and preprocess the MNIST dataset
     - Build a CNN model architecture
     - Train the model (takes ~5-10 minutes)
     - Evaluate model performance
     - Save the trained model to the `models/` directory

3. **Model Architecture**
   ```
   Conv2D (32 filters, 3x3) → MaxPool → 
   Conv2D (64 filters, 3x3) → MaxPool → 
   Conv2D (128 filters, 3x3) → 
   Flatten → Dropout → Dense (128) → Dropout → Dense (10)
   ```

## 🎨 Using the GUI Application

1. **Ensure the model is trained and saved** (complete the notebook first)

2. **Run the GUI application**
   ```powershell
   python digit_recognizer_gui.py
   ```

3. **Using the interface**
   - Draw a digit (0-9) on the black canvas using your mouse
   - Click the "🔍 Predict" button to get the prediction
   - View the predicted digit, confidence score, and probability distribution
   - Click "🗑️ Clear" to reset and draw a new digit

## 🔬 AI Concepts Demonstrated

### Neural Networks
- Multi-layer architecture with forward and backward propagation
- Activation functions (ReLU, Softmax)
- Backpropagation and gradient descent optimization

### Convolutional Neural Networks (CNN)
- **Convolutional Layers**: Extract spatial features from images
- **Pooling Layers**: Reduce spatial dimensions and computational cost
- **Dropout**: Prevent overfitting by randomly dropping neurons
- **Dense Layers**: Final classification based on extracted features

### Training Techniques
- Adam optimizer for adaptive learning
- Categorical cross-entropy loss
- Batch processing for efficient training
- Validation split for model evaluation

## 📈 Model Performance

Expected metrics after training:
- **Training Accuracy**: ~99.5%
- **Validation Accuracy**: ~99.0%
- **Test Accuracy**: ~99.0%

## 🛠️ Technical Details

### Data Preprocessing
1. Reshape images to (28, 28, 1) for CNN input
2. Normalize pixel values to [0, 1] range
3. One-hot encode labels for multi-class classification

### GUI Image Processing
1. Capture drawing from 400x400 canvas
2. Resize to 28x28 pixels (MNIST format)
3. Convert to grayscale and normalize
4. Reshape for model input (1, 28, 28, 1)
5. Make prediction and display results

## 📦 Dependencies

```
tensorflow>=2.10.0
numpy>=1.21.0
matplotlib>=3.4.0
jupyter>=1.0.0
torch>=1.10.0
pillow>=9.0.0
opencv-python>=4.5.0
scikit-learn>=1.0.0
seaborn>=0.11.0
```

## 🐛 Troubleshooting

### Model Not Found Error
- Ensure you've run the Jupyter notebook completely
- Check that `models/` directory contains the saved model files

### GUI Not Responding
- Make sure all dependencies are installed
- Verify TensorFlow is properly installed: `python -c "import tensorflow as tf; print(tf.__version__)"`

### Low Prediction Accuracy
- Draw digits clearly and centered on the canvas
- Make strokes bold and complete
- Try different drawing styles if predictions are incorrect

## 📝 License

This project is for educational purposes.

## 👨‍💻 Author

Nitish Behera


