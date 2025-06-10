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

### Notebook
https://github.com/HamzaMehdi12/Diabetes_Pred/blob/main/Diabetes_Pred_Model/Diabetes_CNN_Detection.ipynb


## Covid19 Forecasting Prediction Model using ARIMAX
The system focuses on forecasting daily new COVID-19 cases using time-series techniques while maintaining temporal relationships through strict datetime indexing. It features automated model configuration via auto_arima, integrated visual diagnostics at multiple stages, and robust preprocessing pipelines. The implementation handles exogenous variables through ARIMAX and provides confidence interval estimates for forecasts. The modular design separates concerns while maintaining interoperability between data processing, modeling, and evaluation components.
## Images
### Arimax Train vs Test
![AriMAX Train vs Test Forecast](https://github.com/user-attachments/assets/9d115593-7e4c-4103-afe7-c508a25aced6)
### Daily Cases
![Daily_Old_New_Deaths_Recovered_And_Total_Cases](https://github.com/user-attachments/assets/db9b4c76-95f7-43da-869e-959728b4a18d)
### Results
![Results](https://github.com/user-attachments/assets/da831ec1-4eb8-4ccb-991b-7196a31a243c)

### Project
Find the project on the link below
https://github.com/HamzaMehdi12/Covid_19-Prediction-Model-using-ARIMAX/blob/main/README.md







