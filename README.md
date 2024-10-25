# Fairness Evaluation on the German Credit Dataset

This project builds and evaluates a Logistic Regression model to predict credit risk using the German Credit Dataset, with an emphasis on ensuring fairness across age groups. It uses a series of fairness techniques to mitigate disparate impacts and assess their effectiveness.

## Project Overview

In the financial sector, regulations often prohibit using sensitive features like age, ethnicity, or marital status in loan approval models to prevent discrimination. This project’s goal is to develop a fair credit risk prediction model by applying various fairness techniques.

### Objectives

- Predict credit risk while ensuring fairness across age groups (< 25 years and ≥ 25 years).
- Evaluate and mitigate disparate impact in model predictions.
- Compare model performance with and without fairness processing.

## Table of Contents

1. **Data Exploration**  
   - Initial analysis, feature selection, and dataset partitioning.
   
2. **Model Training and Evaluation**  
   - Train a Logistic Regression model without fairness preprocessing.
   - Calculate model accuracy, F1 score, and disparate impact.

3. **Fairness Techniques**  
   - **Disparate Impact Removal** using IBM AI Fairness 360.
   - **Reweighing** using Fairlearn.
   - **Equalized Odds** post-processing using ThresholdOptimizer.
   
4. **Evaluation and Comparison**  
   - Assess the disparate impact, accuracy, and F1 score for each fairness technique.

## Datasets

- **Training Data**: `german_credit_training.csv`
- **Test Data**: `german_credit_test.csv`

## Installation

Install the required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn shap aif360 fairlearn
```

## Usage

1. **Data Preparation**  
   - Load and preprocess the dataset by:
     - Handling missing values using imputation.
     - Scaling numerical features and encoding categorical features.
     - Dropping non-relevant columns like `ID` and `age_years`, as `age_groups` is already included as the sensitive attribute.
   
2. **Fairness Processing**  
   - Use various techniques to process the data and assess fairness:
   
     - **Disparate Impact Removal**  
       - Apply IBM's AI Fairness 360 (AIF360) to preprocess the data, reducing disparate impact on the `age_groups` attribute.
       - The `DisparateImpactRemover` is used to transform features without altering labels, making them fairer for model training.

     - **Reweighing**  
       - Use Fairlearn's reweighing method to adjust sample weights based on the sensitive attribute `age_groups`, making the model training more equitable across groups.

     - **Equalized Odds**  
       - Apply the Equalized Odds post-processing technique using Fairlearn’s `ThresholdOptimizer` to balance model predictions across demographic groups in `age_groups`.

3. **Model Evaluation**  
   - Train and evaluate the Logistic Regression model with and without fairness adjustments, comparing the model's performance and fairness:
     - **Metrics Calculated**:
       - **Accuracy** and **F1 Score**: Measure overall prediction quality.
       - **Disparate Impact**: Quantifies bias by assessing if different groups receive different outcomes.
     - **Visualization**:
       - Generate plots to display the distribution of predictions across age groups and the impact of each fairness technique.
   - Store predictions and metrics for each preprocessing technique in separate files to track model performance.

## References
- Kamiran, F., & Calders, T. (2012). Data Preprocessing Techniques for Classification without Discrimination. Knowledge and Information Systems.
- AIF360 documentation: IBM AI Fairness 360
