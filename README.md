# Customer Churn Prediction Algorithm

This project is dedicated to predicting customer churn using a machine learning model. Churn prediction helps businesses understand which customers are likely to stop using their service or product, enabling them to take proactive measures to retain them.

## Project Overview

The primary goal of this project is to build a machine learning model using ANN that predicts whether a customer will churn based on their behavior and other attributes. The project is implemented in a Jupyter Notebook and involves the following steps:
- Data Exploration and Cleaning
- Feature Engineering
- Model Selection and Training
- Model Evaluation
- Prediction on New Data

## Dataset

The dataset used in this project contains customer information, such as demographics, service usage patterns, and customer service interactions. The dataset may be sourced from:
- [Telecom Datasets)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Proprietary Company Data

**Key Features:**
- `customerID`: Unique identifier for each customer
- `tenure`: Number of months the customer has been with the company
- `MonthlyCharges`: The amount charged to the customer each month
- `TotalCharges`: The total amount charged to the customer
- `Contract`: The contract term of the customer (e.g., month-to-month, one year, two years)
- `PaymentMethod`: The method of payment used by the customer
- `Churn`: Whether the customer has churned (Yes/No)

## Installation

To run this project, you need to have the following software installed:

- Python 3.x
- Jupyter Notebook
- Required Python Libraries:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - tensorflow

You can install the required libraries using the following command:

```bash
pip install pandas 
pip install numpy 
pip install scikit-learn 
pip install matplotlib 
pip install seaborn
pip install tensorflow
 ```

# Usage
- Clone this repository to your local machine.
- Open the Jupyter Notebook churn_prediction.ipynb.
- Run the notebook cells sequentially to perform the data analysis, feature engineering, model training, and evaluation.
- Modify the notebook as needed to experiment with different models, features, or datasets.


# Modeling Process
- Data Exploration: Understanding the dataset, identifying missing values, and visualizing data distributions.
- Data Cleaning: Handling missing values, encoding categorical variables, and normalizing numerical features.
- Feature Engineering: Creating new features, selecting important features, and reducing dimensionality if necessary.
- Model Selection: Trying machine learning model  such as ANN.
- Model Training: Training the selected model on the dataset.
- Model Evaluation: Evaluating the model's performance using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC score.
- Prediction: Making predictions on new data or a test set.


# Evaluation
The model's performance is evaluated using the following metrics:

- Accuracy: The proportion of correctly predicted instances.
- Precision: The proportion of true positive predictions among all positive predictions.
- Recall: The proportion of true positive predictions among all actual positives.
- F1-Score: The harmonic mean of precision and recall.


# Results
The final model achieves the following performance on the test set:

- Accuracy: 0.83
- Precision: 0.81
- Recall: 0.91
- F1-Score: 0.85

# Contributing

Contributions are welcome! If you have any ideas, suggestions, or improvements, please feel free to submit a pull request.
