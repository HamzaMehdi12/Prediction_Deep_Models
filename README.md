# Objective
This book is dedicated for prediction models using Deep Learning and Deep Neural Networks


## Diabetes Model using CNN-1D

The model uses a 1D CNN and scikit-learn to detect diabetes, targeting 85%+ accuracy with Adam optimizer.
Dataset: Pima Indians Diabetes (8 features, binary outcome), preprocessed by median imputation, scaling, and reshaping.
Architecture: Conv1D (64/128/256 filters) → Flatten → Dense (512/256/256) → Sigmoid, trained for 200 epochs (batch=32, early stopping).
Performance: 74.03% test accuracy (confusion matrix: TN=43, FP=7, FN=13, TP=14).
Techniques: Median imputation, StandardScaler, Dropout/BatchNorm, early stopping; compared with SVM baseline.
Public dataset, TensorFlow/Keras/scikit-learn implementation.

### Model Accuracy
![alt text](https://github.com/HamzaMehdi12/Diabetes_Pred/blob/main/Diabetes_Pred_Model/TR%20vs%20VAL.png?raw=true)
### Confusion Matrix
![alt text](https://github.com/HamzaMehdi12/Diabetes_Pred/blob/main/Diabetes_Pred_Model/Confusion%20Matrix.png?raw=true)
### Live Example
![alt text](https://github.com/HamzaMehdi12/Diabetes_Pred/blob/main/Diabetes_Pred_Model/Data.png?raw=true)

## Object Detection
