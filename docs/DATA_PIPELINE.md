# Data Pipeline Documentation

## Purpose

This pipeline collects the Iris dataset, preprocesses the data,
creates new features, and validates the final dataset.

## Pipeline Stages

### 1. Data Collection
- Input: Iris dataset from scikit-learn
- Output: `data/raw/iris_raw.csv`
- Records collected: 150

### 2. Data Preprocessing
- Input: `data/raw/iris_raw.csv`
- Removes duplicate rows
- Handles missing numeric values using median
- Removes rows with missing target values
- Output: `data/processed/iris_preprocessed.csv`

### 3. Feature Engineering
- Input: `data/processed/iris_preprocessed.csv`
- Creates sepal area
- Creates petal area
- Creates sepal-to-petal length ratio
- Creates petal length categories
- Output: `data/processed/iris_features.csv`

### 4. Data Validation
- Checks required columns
- Checks for null values
- Checks valid species values
- Checks numeric value ranges
- Pipeline stops if validation fails

## Pipeline Flow

```text
Iris Dataset
     |
     v
Data Collection
     |
     v
Raw Data
     |
     v
Preprocessing
     |
     v
Processed Data
     |
     v
Feature Engineering
     |
     v
Feature Data
     |
     v
Data Validation
     |
     v
Validated Dataset