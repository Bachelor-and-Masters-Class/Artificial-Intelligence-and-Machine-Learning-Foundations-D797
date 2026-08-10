**Overview**

This repository contains my completed submission for the WGU course **D797: Artificial Intelligence and Machine Learning Foundations**.

The project applies both **unsupervised** and **supervised** machine learning to a real-world public health dataset: Maryland's COVID-19 case and outcome data. The pipeline cleans the raw data, normalizes numeric features to a common scale, segments observations into behavioral clusters using K-Means, and then trains a Random Forest regressor to predict a continuous outcome, evaluated with root mean squared error.

The assignment demonstrates the foundational machine learning workflow end to end: preprocessing, unsupervised pattern discovery, supervised prediction, quantitative evaluation, and model persistence.

**Scenario**

Public health agencies collect large volumes of jurisdiction-level pandemic data. The raw dataset contains numeric measures on widely different scales alongside identifier columns that carry no predictive meaning.

The objective is to prepare this data correctly, identify natural groupings within it, and build a predictive model whose performance can be measured and whose artifact can be reused without retraining.

**Project Objectives**

-Load and clean a real-world public health dataset

-Distinguish identifier columns from genuine predictive features

-Normalize numeric features so that no single variable dominates by scale

-Apply unsupervised clustering to discover structure in the data

-Train a supervised ensemble regression model

-Evaluate model performance using an interpretable error metric

-Persist the trained model as a reusable artifact

**Technical Implementation**

**Data Loading and Cleaning**
The Maryland COVID-19 dataset is read from Excel with pandas. Rows containing missing values are removed to ensure both the clustering and regression stages operate on complete records.

**Feature Selection**
Identifier columns such as OBJECTID are explicitly excluded from the feature set. Including record identifiers would introduce meaningless signal and inflate apparent model performance.

**Normalization: MinMaxScaler**
All retained numeric features are scaled into the 0-1 range. This is essential before K-Means, which relies on Euclidean distance and would otherwise be dominated by whichever feature happens to have the largest raw magnitude.

**Unsupervised Learning: K-Means Clustering**
K-Means partitions the scaled observations into three clusters with a fixed random seed and multiple initializations to guard against poor centroid placement. The resulting cluster assignment is attached back to the dataset as a derived feature.

**Supervised Learning: Random Forest Regression**
A Random Forest regressor with 100 estimators is trained on an 80/20 split. The ensemble approach was selected because it handles non-linear relationships and feature interactions without requiring manual feature engineering, and is resistant to overfitting relative to a single decision tree.

**Evaluation**
Model performance is measured with root mean squared error, which reports error in the same units as the target and penalizes large individual errors more heavily than mean absolute error.

**Model Persistence**
The fitted model is serialized with joblib so it can be reloaded and applied to new data without retraining.

**Skills Demonstrated**

-Supervised and unsupervised machine learning fundamentals

-Data cleaning and missing value handling

-Feature selection and identifier exclusion

-Feature scaling and normalization

-K-Means clustering and cluster-based feature engineering

-Random Forest ensemble regression

-Regression evaluation with RMSE

-Model serialization and artifact management

**How to Run**

```
python src/covid_model.py
```

The script reads the dataset from the data directory, prints the RMSE, and writes the trained model to disk.

**Repository Contents**

-`src/covid_model.py`: Preprocessing, clustering, model training, evaluation, and persistence

-`data/MD_COVID_19_dataset.xlsx`: Maryland COVID-19 source dataset
