# handson-10-MachineLearning-with-MLlib.

#  Customer Churn Prediction with MLlib

This project uses Apache Spark MLlib to predict customer churn based on structured customer data. You will preprocess data, train classification models, perform feature selection, and tune hyperparameters using cross-validation.Build and compare machine learning models using PySpark to predict whether a customer will churn based on their service usage and subscription features.

---

##  Dataset

The dataset used is `customer_churn.csv`, which includes features like:

- `gender`, `SeniorCitizen`, `tenure`, `PhoneService`, `InternetService`, `MonthlyCharges`, `TotalCharges`, `Churn` (label), etc.

---

### Code Explanation
## 🔹 Task 1: Data Preprocessing and Feature Engineering

**Objective**:  
Prepare the raw customer churn dataset for machine learning by handling missing values, encoding categorical variables, and assembling features into a vector.

**Steps**:
- Fill missing values in the `TotalCharges` column with `0`.
- Convert categorical columns (`gender`, `PhoneService`, `InternetService`) to numeric using `StringIndexer`.
- Apply `OneHotEncoder` to these indexed columns.
- Combine numerical and encoded categorical features using `VectorAssembler`.
---
## 🔹 Task 2: Train and Evaluate a Logistic Regression Model

**Objective**:  
Train a logistic regression model to classify churn and evaluate its performance using AUC.

**Steps**:
- Use `StringIndexer` to convert `Churn` (Yes/No) into binary `label` (1/0).
- Split the dataset into 80% training and 20% test data.
- Train a `LogisticRegression` model on the training set.
- Predict on the test set and compute AUC with `BinaryClassificationEvaluator`.
---
## 🔹 Task 3: Feature Selection using Chi-Square Test

**Objective**:  
Identify the top 5 most relevant features using a statistical Chi-Square test.

**Steps**:
- Ensure `label` column is created from the `Churn` column.
- Use `ChiSqSelector` to select top 5 features most correlated with churn.
- Output the reduced dataset showing `selectedFeatures` and `label`.
---
## 🔹 Task 4: Hyperparameter Tuning and Model Comparison

**Objective**:  
Train and compare multiple machine learning models using cross-validation and select the best-performing one based on AUC.

**Steps**:
- Define four models:  
  - `LogisticRegression`  
  - `DecisionTreeClassifier`  
  - `RandomForestClassifier`  
  - `GBTClassifier` (Gradient Boosted Trees)
- Build hyperparameter grids for each model.
- Use 5-fold `CrossValidator` to train and evaluate models.
- Compare AUC scores and identify the best model.
---

### Outputs
### Task 1: Data Preprocessing and Feature Engineering

**Objective:**  
Clean the dataset and prepare features for ML algorithms.

**Steps:**
1. Fill missing values in `TotalCharges` with 0.
2. Encode categorical features using `StringIndexer` and `OneHotEncoder`.
3. Assemble numeric and encoded features into a single feature vector with `VectorAssembler`.

**Code Output:**

```
+------------------------------------------------+----------+
|features                                        |ChurnIndex|
+------------------------------------------------+----------+
|[11.0,64.0,778.36,1.0,0.0,1.0,0.0,1.0,0.0,0.0]  |0.0       |
|[6.0,73.14,478.43,0.0,1.0,1.0,0.0,0.0,0.0,1.0]  |1.0       |
|[70.0,73.43,5544.84,0.0,1.0,0.0,1.0,0.0,1.0,0.0]|1.0       |
|[51.0,81.15,4086.94,0.0,1.0,1.0,0.0,0.0,0.0,1.0]|0.0       |
|[4.0,100.39,399.31,0.0,1.0,1.0,0.0,1.0,0.0,0.0] |1.0       |
+------------------------------------------------+----------+
```
---

### Task 2: Train and Evaluate Logistic Regression Model

**Objective:**  
Train a logistic regression model and evaluate it using AUC (Area Under ROC Curve).

**Steps:**
1. Split dataset into training and test sets (80/20).
2. Train a logistic regression model.
3. Use `BinaryClassificationEvaluator` to evaluate.

**Code Output Example:**
```
Logistic Regression AUC: 0.7074457690052764
```

---

###  Task 3: Feature Selection using Chi-Square Test

**Objective:**  
Select the top 5 most important features using Chi-Square feature selection.

**Steps:**
1. Use `ChiSqSelector` to rank and select top 5 features.
2. Print the selected feature vectors.

**Code Output Example:**
```
+----------------------+----------+
|selectedFeatures      |ChurnIndex|
+----------------------+----------+
|[11.0,1.0,0.0,1.0,0.0]|0.0       |
|[6.0,0.0,1.0,0.0,0.0] |1.0       |
|[70.0,0.0,1.0,0.0,1.0]|1.0       |
|[51.0,0.0,1.0,0.0,0.0]|0.0       |
|[4.0,0.0,1.0,1.0,0.0] |1.0       |
+----------------------+----------+

```

---

### Task 4: Hyperparameter Tuning and Model Comparison

**Objective:**  
Use CrossValidator to tune models and compare their AUC performance.

**Models Used:**
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosted Trees (GBT)

**Steps:**
1. Define models and parameter grids.
2. Use `CrossValidator` for 5-fold cross-validation.
3. Evaluate and print best model results.

**Code Output Example:**
```
LogisticRegression AUC: 0.7066640609732265
DecisionTree AUC: 0.6608364275942935
RandomForest AUC: 0.7809263240179792
GBT AUC: 0.7639241743208913
Best Model: RandomForest with AUC: 0.7809263240179792
```
---

##  Execution Instructions

### 1. Prerequisites

- Apache Spark installed
- Python environment with `pyspark` installed
- `customer_churn.csv` placed in the project directory

### 2. Run the Project

```bash
spark-submit churn_prediction.py
```
