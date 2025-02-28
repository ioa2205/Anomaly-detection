# Anomaly Detection in Time-Series Data

## **Overview**
This project focuses on detecting anomalies in time-series data, specifically temperature readings from a machine. The goal is to identify unusual patterns that may indicate system failures. The model used is **Isolation Forest**, which is trained on normal data and flags anomalies when a data point significantly deviates from the expected pattern.

## **Dataset**
The dataset is taken from **Numenta Anomaly Benchmark (NAB)** and includes machine temperature readings over time. Labeled anomalies are provided for evaluation.

## **Technologies Used**
- **Python** for data processing and model training
- **Scikit-learn** for implementing Isolation Forest
- **Queue** for real-time data streaming
- **Joblib** for saving and loading the trained model
- **NumPy & Pandas** for data manipulation


## **Model Usage**
The trained model is saved as `isolation_forest_model.pkl`. To use it:
```python
import joblib
import numpy as np

# Load the model
model = joblib.load("isolation_forest_model.pkl")

def predict_anomaly(features):
    prediction = model.predict(np.array([features]))[0]
    return "Anomaly" if prediction == -1 else "Normal"

# Example usage
test_features = [30, 27.5, 2.1]  # [Temperature, Rolling Mean, Rolling Std]
print(predict_anomaly(test_features))
```
## **Evaluation**

True Anomalies: 567 out of 4538 (12.49%)

Precision: 0.9933

Recall: 0.7866

F1-score: 0.8780

ROC-AUC: 0.8929

Specificity: 0.9992

Confusion Matrix:

      True Negatives (TN): 3968
  
      False Positives (FP): 3
  
      False Negatives (FN): 121
  
      True Positives (TP): 446

![image](https://github.com/user-attachments/assets/78a7e131-1d16-406d-9156-4ee0728f2be0)
## **Future Improvements**
- Implement **Deep Learning models** (LSTM, Autoencoders) for better anomaly detection.
- Improve **feature engineering** to capture more complex patterns.
- Deploy as a **real-time monitoring system** with dashboards.

## **License**
This project is licensed under the MIT License.

