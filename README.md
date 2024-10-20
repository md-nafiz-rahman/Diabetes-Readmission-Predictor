# Diabetes Readmission Predictor

This repository contains a data science project aimed at predicting hospital readmission for diabetic patients using various machine learning techniques. The project focuses on feature engineering, model evaluation, and clustering to identify key factors impacting readmission rates.

## Project Overview

The dataset used represents 10 years of clinical care data (1999-2008) from 130 US hospitals, focusing on diabetic patients. This project explores various predictors, such as age, race, gender, number of hospital visits, and medical procedures, to build a model capable of predicting whether a patient will be readmitted to the hospital.

### Key Objectives:
1. **Data Cleansing and Feature Engineering**: Cleaning and transforming the data by dealing with missing values, outliers, and skewness in features.
2. **Classification Models**: Implementing RandomForest and Logistic Regression models to predict patient readmission.
3. **Balancing the Dataset**: Using oversampling techniques to balance the minority class (readmitted patients) and improve model accuracy.
4. **Clustering with K-Means**: Applying the K-Means clustering algorithm to group patients based on their medical conditions and hospital visits, providing insights for healthcare decision-making.

## Models Implemented

1. **Random Forest Classifier**:
   - Provides a high level of accuracy and is well-suited for handling high-dimensional data.
   - Trained on both the original and balanced datasets.
   - Evaluated using cross-validation, confusion matrix, and classification report.
   
2. **Logistic Regression**:
   - Simple baseline model for comparison.
   - Evaluated using precision, recall, and F1 score.

3. **K-Means Clustering**:
   - Used for grouping patients based on characteristics such as age, number of hospital visits, and medical procedures.
   - Optimal number of clusters determined using the Elbow Method and visualized through PCA.

## Feature Engineering and Data Processing

- **Handling Missing Values**: Replacing missing values and removing columns with more than 50% missing data.
- **Outlier Removal**: Using z-scores and IQR methods to remove outliers from the dataset.
- **Log Transformation**: Applied to skewed numerical features to normalize the data.
- **Dummy Variables**: Created for categorical columns such as race, gender, and admission/discharge IDs.
- **Correlation Analysis**: Visualized to understand relationships between features and target variables.
