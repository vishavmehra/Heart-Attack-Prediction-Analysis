# Heart Attack Prediction Analysis

This project analyzes and predicts the likelihood of heart attacks using machine learning models. By utilizing the Pycaret library, we streamline the process of data preprocessing, exploratory data analysis (EDA), and model training. This analysis is done on a dataset obtained from Kaggle that includes a variety of health indicators linked to heart disease risk.

## Project Overview

The main objective is to leverage machine learning to predict heart attack risks based on patient data. We explore and process the dataset, visualize important patterns, and experiment with various models to select the one with the highest predictive performance. This analysis provides insights into which health metrics have the greatest influence on heart attack risk, making it a valuable tool for healthcare analytics.

## Features

- **Data Cleaning and Preprocessing:** Handles missing data, scales features, and manages outliers for optimal model performance.
- **Exploratory Data Analysis (EDA):** Uses visualizations to highlight significant relationships and trends within the data.
- **Automated Model Selection and Comparison:** Employs Pycaret to train multiple models and compare their accuracy, efficiency, and interpretability.

## Technologies Used

- **Python:** The main programming language for data processing and model training.
- **Pycaret:** Streamlines the machine learning pipeline, enabling rapid model comparison and tuning.
- **Google Colab:** Provides an accessible, cloud-based Jupyter environment for running this analysis without requiring extensive local setup.

---

## Codebase Structure

The code is structured within a Jupyter notebook (`heart_attack_analysis.ipynb`) and organized to follow a sequential analysis workflow. Below is an outline of each section of the notebook:

### 1. **Setup: Importing Libraries and Dependencies**

   - Libraries such as `pandas`, `numpy`, `matplotlib`, and `seaborn` are imported to handle data processing and visualization.
   - Pycaret is imported to simplify and automate machine learning model training, allowing for quick model comparisons.

### 2. **Data Loading and Inspection**

   - Loads the heart attack dataset from Kaggle, detailing features like age, cholesterol, blood pressure, and other key clinical indicators.
   - Displays the first few rows of the dataset to get an initial understanding of its structure, columns, and any potential missing values.

### 3. **Data Preprocessing**

   - **Handling Missing Values:** Missing data is handled through imputation strategies that help preserve the integrity of the dataset.
   - **Outlier Detection and Removal:** Detects outliers in key features that might skew results and either removes or adjusts these values to maintain model accuracy.
   - **Feature Scaling:** Normalizes data to bring features into a common scale, especially useful when dealing with algorithms sensitive to feature magnitude.

### 4. **Exploratory Data Analysis (EDA)**

   - **Statistical Summary:** Summarizes each feature with measures like mean, median, min, max, and standard deviation to understand data distribution.
   - **Visualizations:**
     - **Histograms and Box Plots:** Show the distribution and range of features, highlighting any anomalies.
     - **Correlation Heatmap:** Visualizes relationships between features to identify which attributes might most influence heart attack prediction.
     - **Pair Plots:** Displays scatter plots for each feature pair, allowing insight into linear relationships and potential feature importance.

### 5. **Feature Engineering (if necessary)**

   - In cases where additional insights can be derived, new features may be created. This could include transformations or combinations of existing features (e.g., cholesterol-to-age ratio).

### 6. **Model Training and Comparison (using Pycaret)**

   - **Setting Up Pycaret Environment:** Defines the target variable (heart attack occurrence) and feature variables, configuring Pycaret to handle data preprocessing, splitting, and modeling in one step.
   - **Model Selection and Training:** Pycaret evaluates multiple classification models, including:
     - **Logistic Regression**
     - **Decision Trees**
     - **Random Forests**
     - **K-Nearest Neighbors (KNN)**
     - **Gradient Boosting Machines**
   - **Model Comparison:** Pycaret ranks models based on metrics like accuracy, precision, recall, F1-score, and AUC (Area Under the Curve), providing a summary table of model performance.
   - **Best Model Selection:** Selects the model with the highest accuracy or other preferred metric for final prediction and validation.

### 7. **Evaluation and Interpretation of Results**

   - **Best Model Analysis:** Provides a more detailed evaluation of the top-performing model's metrics, including confusion matrix, ROC curve, and feature importance.
   - **Feature Importance Visualization:** Highlights the most influential features, such as age, cholesterol level, and blood pressure, giving insights into the factors most associated with heart disease risk.

### 8. **Conclusions and Key Findings**

   - Summarizes the predictive insights derived from the analysis.
   - Highlights the most influential health indicators that could be prioritized in healthcare settings.
   - Provides recommendations for further model optimization, like hyperparameter tuning or additional feature engineering.

### 9. **Future Work and Improvements**

   - Suggestions for potential improvements include experimenting with more complex models, such as ensemble methods, or gathering more data to enhance model accuracy.

---

## Getting Started

To replicate or extend the analysis:
1. **Clone the repository:** `git clone https://github.com/vishavmehra/Heart-Attack-Prediction-Analysis.git`
2. **Open the notebook:** Use Google Colab or Jupyter Notebook to run `heart_attack_file.ipynb`.
3. **Execute Cells Sequentially:** The notebook is organized to run step-by-step, from data loading to final conclusions.

## Dataset

The dataset used for this analysis can be downloaded from Kaggle and contains anonymized patient health data.

---

## Results Summary

- **Best Model Performance:** The highest-performing model's metrics, including accuracy and F1-score, are detailed in the notebook.
- **Feature Importance:** The analysis provides a ranked list of features, helping identify which health indicators contribute most to heart disease risk.
- **Predictive Insights:** The project highlights potential health metrics that healthcare professionals may want to monitor more closely for heart disease risk.

## Contributing

Contributions are encouraged! If you have suggestions, improvements, or additional insights, feel free to fork the repository and submit a pull request. Issues can also be opened for discussion.

