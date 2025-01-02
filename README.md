# Human Activity Recognition Using Smartphone Sensors

## Overview
This project implements Human Activity Recognition (HAR) using smartphone sensor data to classify six daily activities: Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, and Laying. The project utilizes accelerometer and gyroscope data collected from smartphones mounted on participants' waists.

## Project Structure
- Data preprocessing and feature engineering
- Exploratory Data Analysis (EDA)
- Multiple modeling approaches:
  - Classical Machine Learning
  - Dimensionality Reduction
  - Deep Learning with LSTM

## Dataset Description
- **Source**: UCI HAR Dataset
- **Participants**: 30 subjects
- **Sampling Rate**: 50Hz
- **Window Size**: 2.56 seconds (128 readings) with 50% overlap
- **Features**: 561 feature vector with time and frequency domain variables
- **Activities**: 6 activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING)

## Exploratory Data Analysis

### Key Findings
1. Activity Distribution Analysis:
   - Nearly equal distribution of activities in the dataset
   - Walking upstairs takes approximately 10% longer than walking downstairs
![newplot](https://github.com/user-attachments/assets/17c82a0c-6ea5-4bcc-a1ac-3e9534e3d267)

2. Signal Analysis:
   - Clear separation between stationary and moving activities
![937d01a1-5eef-4799-9af8-ef9d3a0d55ea](https://github.com/user-attachments/assets/0ec9283d-327e-443d-8f10-91fc379c190b)
   - Distinct patterns in acceleration magnitude for different activities
     ![eb5c557a-e4d9-4fdb-b7e5-b507395aeda9](https://github.com/user-attachments/assets/741c9463-1345-43a4-a694-fb9a43f27652)
   - Gravity acceleration components help distinguish between activities
![4bc302a5-6539-4fbf-82f2-afbe00e3271a](https://github.com/user-attachments/assets/0654782a-0b8e-41e3-99fd-d534cfab6c4a)
![d1ab194d-15b4-4e54-9e91-303a4579a49b](https://github.com/user-attachments/assets/8e5ef239-d99b-4279-ac3c-768c8628bee9)

3. Feature Analysis:
   - Correlation analysis revealed redundant features
    ![174b5957-74f1-4e5d-8280-e4ebaa1cb96c](https://github.com/user-attachments/assets/8ca451df-15b7-4971-b2b5-8490cc8d5127)
   - T-SNE visualization showed clear clustering of activities except for "Standing" and "Sitting"
![793293c9-b39b-4d4f-b9ac-b94e42d63716](https://github.com/user-attachments/assets/533ef734-2b3d-4cec-8993-421ce05a617b)

### Raw Signal Visualization
- Plotted body acceleration (X, Y, Z axes)![a80bb4b2-b2c5-4e0b-9f87-017b60f954e4](https://github.com/user-attachments/assets/0de29943-7172-43a0-a151-b24c51981853)

- Analyzed gyroscope data patterns![94216dd3-88b9-4a03-9a82-24f0a6887365](https://github.com/user-attachments/assets/116d35d5-3644-41eb-b8b1-8840bc826bac)

- Created correlation matrices for features

## Modeling Approaches

### 1. Classical Machine Learning Models
Results before dimensionality reduction:

| Model | Accuracy |
|-------|----------|
| Support Vector Machine (poly) | 96.30% |
| Logistic Regression | 95.86% |
| XGBoost | 95.05% |
| Random Forest | 91.96% |
| K-Nearest Neighbours | 90.23% |
| DecisionTree | 86.39% |

### 2. Dimensionality Reduction

#### PCA Results (95% variance retained):
| Model | Accuracy |
|-------|----------|
| SVM (rbf) | 95.76% |
| Logistic Regression | 94.13% |
| XGBoost | 92.57% |
| Random Forest | 91.21% |
| KNN | 87.89% |

#### SelectKBest Results:
| Model | Accuracy |
|-------|----------|
| SVM (rbf) | 87.00% |
| Logistic Regression | 86.39% |
| Random Forest | 85.14% |
| XGBoost | 83.41% |
| KNN | 82.25% |

### 3. Deep Learning LSTM Models
| Architecture | Loss | Accuracy |
|--------------|------|----------|
| 1-Layer (32 neurons) | 0.50 | 90% |
| 2-Layer (48, 32 neurons) | 0.34 | 91% |
| 2-Layer (64, 48 neurons) | 0.32 | 92% |

## Feature Importance
Top contributing features:
1. tBodyGyroJerk-entropy()-X (0.081778)
2. fBodyGyro-entropy()-X (0.051239)
3. fBodyAcc-entropy()-X (0.016966)
4. tGravityAcc-mean()-Y (0.009841)
5. fBodyBodyAccJerkMag-entropy() (0.009026)

## Key Conclusions

1. **Best Models**:
   - SVM with polynomial kernel achieved highest accuracy (96.3%)
   - Logistic Regression offers good balance of accuracy (95.86%) and training speed
   - XGBoost performs well (95.05%) but requires significantly more training time

2. **Dimensionality Reduction**:
   - PCA with 95% variance threshold effectively reduced dimensions while maintaining high accuracy
   - SVM with RBF kernel performed best on PCA-reduced features (95.76%)

3. **Deep Learning Performance**:
   - LSTM models achieved good accuracy (92%) using raw data without feature engineering
   - Increased LSTM layers and neurons improved performance
   - Decreasing cross-entropy loss correlated with increasing accuracy

## Implementation Details

### Data Processing
```python
# Features for raw signals
SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z"
]
```

### Model Parameters
- **SVM**: 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'kernel': ['linear', 'poly', 'rbf']
- **Logistic Regression**: 'C':[0.01, 0.1, 1, 10, 20, 30], 'penalty':['l2','l1']
- **XGBoost**: 'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
- **LSTM**: batch_size=16, epochs=30, dropout=0.5
- **DecisionTrees**: 'max_depth':np.arange(3,10,2)
- **RandomForest**: 'n_estimators': np.arange(10,201,20), 'max_depth':np.arange(3,15,2)
- **K-Nearest Neighbours**: 'n_neighbors': [3, 5, 7, 9]

## Requirements
- Python 3.7+
- scikit-learn
- TensorFlow/Keras
- XGBoost
- pandas
- numpy
- matplotlib
- seaborn

