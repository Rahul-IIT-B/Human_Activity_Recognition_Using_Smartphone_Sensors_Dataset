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

2. Signal Analysis:
   - Clear separation between stationary and moving activities
   - Distinct patterns in acceleration magnitude for different activities
   - Gravity acceleration components help distinguish between activities

3. Feature Analysis:
   - Correlation analysis revealed redundant features
   - T-SNE visualization showed clear clustering of activities except for "Standing" and "Sitting"

### Raw Signal Visualization
- Plotted body acceleration (X, Y, Z axes)
- Analyzed gyroscope data patterns
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
- **SVM**: C=10, gamma='scale', kernel='poly'
- **Logistic Regression**: C=1, penalty='l2'
- **XGBoost**: learning_rate=0.2, max_depth=3, n_estimators=200
- **LSTM**: batch_size=16, epochs=30, dropout=0.5

## Future Improvements
1. Feature selection optimization
2. Hybrid model approaches
3. Real-time prediction implementation
4. Cross-validation with different window sizes
5. Ensemble method exploration

## Requirements
- Python 3.7+
- scikit-learn
- TensorFlow/Keras
- XGBoost
- pandas
- numpy
- matplotlib
- seaborn

