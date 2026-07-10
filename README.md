# Credit Card Fraud Detection using Machine Learning

An end-to-end machine learning project to detect fraudulent credit card transactions using classification algorithms, resampling techniques, and ensemble learning.

## Overview

Credit card fraud detection is a challenging machine learning problem because fraudulent transactions represent only a very small percentage of total transactions.

This project builds a complete ML pipeline to handle class imbalance, train multiple classification models, compare their performance, and identify the best approach for fraud detection.

## Dataset

**Source:** Kaggle Credit Card Fraud Detection Dataset

Dataset details:

* Total Transactions: 284,807
* Fraud Cases: 492
* Features:

  * V1–V28 (PCA transformed features)
  * Transaction Time
  * Transaction Amount
  * Class (Target: Fraud / Legitimate)

## Project Workflow

```
Data Collection
        |
        ↓
Exploratory Data Analysis
        |
        ↓
Data Preprocessing
        |
        ↓
Handling Class Imbalance
        |
        ↓
Model Training
        |
        ↓
Model Evaluation
        |
        ↓
Fraud Prediction
```

## Key Features

* Performed exploratory data analysis on transaction patterns
* Handled highly imbalanced data using:

  * SMOTE
  * Random Undersampling
  * NearMiss
  * KMeans-based undersampling
* Built and compared multiple machine learning models:

  * Logistic Regression
  * Random Forest
  * K-Nearest Neighbors
  * Neural Network (MLP)
  * Voting Classifier
* Optimized classification thresholds to improve fraud detection performance
* Evaluated models using:

  * Precision
  * Recall
  * F1-score
  * PR-AUC

## Model Performance

| Model             | F1 Score | Precision | Recall |
| ----------------- | -------- | --------- | ------ |
| Neural Network    | 85.25%   | 90.70%    | 80.41% |
| Random Forest     | 84.69%   | 83.84%    | 85.57% |
| KNN               | 87.05%   | 87.50%    | 86.60% |
| Voting Classifier | 86.73%   | 85.86%    | 87.63% |

## Project Structure

```
Credit-Card-Fraud-Detection/

├── notebooks/
│   └── credit_fraud.ipynb        # Data analysis and experiments

├── scripts/
│   ├── credit_fraud_train.py     # Model training
│   ├── credit_fraud_test.py      # Model evaluation
│   ├── credit_fraud_utils_data.py
│   ├── credit_fraud_utils_eval.py
│   ├── kmeans_undersampler.py
│   └── cli_args.py

├── summary/
│   └── project_report.pdf

├── saved_models/
│   └── Trained models generated after training

├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/nitikachauhann/credit-card-fraud-detection-ml.git
cd credit-card-fraud-detection-ml
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn
* Matplotlib
* Seaborn
* Jupyter Notebook

## Future Improvements

* Deploy the model using FastAPI
* Add real-time transaction prediction API
* Add explainable AI using SHAP
* Build an interactive fraud monitoring dashboard

## Author

Nitika Chauhan
