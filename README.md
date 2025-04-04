Objective
Implement a diabetes detection model using a 1D CNN layer and scikit-learn classifiers.

Aim for an accuracy of 85%+ using the Adam optimizer.

Dataset
Source: Pima Indians Diabetes dataset from UCI (URL provided).

Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age.

Target: Outcome (binary classification: diabetic or non-diabetic).

Preprocessing:

Replaced zero values with median for relevant columns (Glucose, BloodPressure, etc.).

Split into training (90%) and testing (10%) sets.

StandardScaler used for feature scaling.

Reshaped data for Conv1D input.

Model Architecture
Type: Sequential CNN model.

Layers:

Multiple Conv1D layers (64, 128, 256 filters) with ReLU activation, BatchNormalization, and Dropout.

Flatten layer followed by Dense layers (512, 256, 256 neurons) with ReLU activation, BatchNormalization, and Dropout.

Final Dense layer with sigmoid activation for binary classification.

Optimizer: Adam with learning rate scheduling (initial rate = 0.0005, exponential decay).

Loss Function: Binary cross-entropy.

Metrics: Accuracy and AUC.

Training
Epochs: 200 with early stopping (patience = 20).

Batch Size: 32.

Validation Data: 10% of the training set.

Results:

Training accuracy: ~74% (varies by epoch).

# Validation accuracy: ~74% (best achieved).

## Evaluation
Accuracy: 74.03% on the test set.

## Confusion Matrix:

True Negatives: 43

False Positives: 7

False Negatives: 13

True Positives: 14.

## Visualization
Plotted training/validation accuracy and loss over epochs.

## Key Techniques
Handling Missing Values: Zero values replaced with median.

Feature Scaling: StandardScaler applied.

Regularization: Dropout and BatchNormalization to prevent overfitting.

Early Stopping: To halt training if validation loss does not improve.

## Additional Notes
The notebook includes references to TensorFlow, Keras, and scikit-learn documentation.

The model is compared with an SVM (SVC) for baseline performance (not detailed in the provided code).

The dataset is publicly available and commonly used for binary classification tasks.

## Ouput Images
### Model Accuracy
![alt text](https://github.com/HamzaMehdi12/Diabetes_Pred/blob/main/TR%20vs%20VAL.png?raw=true)
