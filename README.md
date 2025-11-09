# ❤️ Heart Disease Prediction: A Data Cleaning & Classification Project

This project is an end-to-end machine learning workflow to predict the presence of heart disease based on a set of medical attributes.

A key part of this project is the **data cleaning and validation** process. The popular 1025-row `heart.csv` dataset from Kaggle, which often leads to misleading 100% accuracy scores, was found to contain **722 duplicate rows**.

This notebook demonstrates the critical importance of identifying and removing this leaked data to build a realistic, trustworthy, and generalizable predictive model.

## 📊 Dataset

The dataset used is a popular version of the Cleveland Clinic Foundation heart disease dataset available on Kaggle.

* **Source:** [Heart Disease Dataset on Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) (Note: We use the 1025-row version and de-duplicate it).
* **Target Variable:** `target` (1 = Heart Disease, 0 = No Heart Disease)

---

## ⚙️ Project Workflow

1.  **Data Cleaning (The Critical Fix):**
    * Loaded the 1025-row dataset and immediately identified **722 duplicate rows**.
    * Removed all duplicates, resulting in a clean, 303-row dataset, which is the correct, original Cleveland dataset. This step is essential to prevent data leakage and fix the "100% accuracy" flaw.

2.  **Exploratory Data Analysis (EDA):**
    * Checked the target variable balance, finding it to be very well-balanced (approx. 54% '1's and 46% '0's).
    * Visualized feature distributions (`age`, `chol`, `thalach`, etc.) using histograms and box plots to understand how they differ between positive and negative cases.
    * Plotted a correlation heatmap to identify the strongest linear predictors (`cp`, `thalach`, `slope`).

3.  **Preprocessing & Train-Test Split:**
    * The clean, de-duplicated data was split into an 80% training set and a 20% test set, using `stratify=y` to maintain the class balance.
    * A `StandardScaler` was fit *only* on the training data and then used to scale both the train and test sets to prevent data leakage.

4.  **Baseline Modeling:**
    * Trained four baseline models (Logistic Regression, Decision Tree, Random Forest, AdaBoost) on the scaled data.
    * This established a realistic performance baseline (e.g., Logistic Regression ROC AUC of ~0.87), proving the 100% accuracy was flawed.

5.  **Hyperparameter Tuning:**
    * `RandomizedSearchCV` was used to tune the `RandomForestClassifier` (the most promising model), optimizing for the **ROC AUC** score over 5 cross-validation folds.
    * This process found an optimal set of parameters (`max_depth: 5`, `max_features: 'log2'`, etc.) with a strong CV ROC AUC of **0.9125**.

6.  **Final Evaluation & Explainability:**
    * The final tuned model was evaluated on the unseen test set, achieving a **Test Set ROC AUC of 0.88**.
    * **Feature Importance** was extracted from the tuned model, confirming that `ca` (number of major vessels), `cp` (chest pain type), and `thalach` (max heart rate) were the most important predictors.

---

## 🚀 Key Findings & Results

The original project's 100% accuracy was confirmed to be illusory and a direct result of massive data duplication.

By cleaning the data and building a robust pipeline, we achieved a realistic and strong model.

**Final Tuned Random Forest (Test Set Performance):**
* **Test Set Accuracy:** 78.69%
* **Test Set ROC AUC:** 0.8755
* **Test Set Precision (Disease):** 0.79
* **Test Set Recall (Disease):** 0.82

This project successfully demonstrates a trustworthy, end-to-end process for building an interpretable heart disease prediction model.




---

## 🛠️ Tools and Libraries Used
* Pandas
* NumPy
* Matplotlib & Seaborn
* Scikit-learn (StandardScaler, train_test_split, RandomizedSearchCV, classification_report, roc_auc_score, etc.)
* AdaBoost
