Telecommunication Customer Churn Prediction

Project Overview

This project aims to predict customer churn for a telecommunications
company using machine learning. By identifying customers at risk of
churning, the company can implement targeted retention strategies,
thereby reducing customer attrition and improving customer lifetime
value. The analysis involves Exploratory Data Analysis (EDA), feature
engineering, model training, and performance evaluation, with a focus on
optimizing for recall for the minority ‘Churn’ class.

Dataset

The dataset, Telco.csv, contains customer information from a
telecommunications company, including various demographic details,
services subscribed to, monthly charges, total charges, and a ‘Churn’
column indicating whether the customer churned or not.

Methodology

1. Data Loading and Initial Inspection

-   Loaded the Telco.csv file into a Pandas DataFrame.
-   Performed initial checks for missing values and duplicates.
-   Converted TotalCharges to a numeric type, handling empty strings and
    NaN values by dropping corresponding rows.

2. Exploratory Data Analysis (EDA)

-   Distribution of Churn Classes: Analyzed the balance of ‘Churn’
    vs. ‘No Churn’ customers, revealing a class imbalance.
-   Qualitative Feature Analysis: Visualized the distribution of
    categorical features against ‘Churn’ using count plots and
    probability heatmaps. Key observations included:
    -   SeniorCitizen: Higher churn probability for senior citizens.
    -   Partner and Dependents: Customers without partners or dependents
        showed higher churn rates.
    -   InternetService: Fiber optic users had a significantly higher
        churn rate.
    -   OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
        StreamingTV, StreamingMovies: Absence of these services
        correlated with higher churn.
    -   Contract: Month-to-month contracts showed much higher churn.
    -   PaperlessBilling: Higher churn rate for customers using
        paperless billing.
    -   PaymentMethod: Electronic check users had a significantly higher
        churn probability.
-   Quantitative Feature Analysis: Examined numerical features (tenure,
    MonthlyCharges, TotalCharges) distribution against ‘Churn’ using KDE
    plots and box plots.

3. Feature Selection

-   Chi-Squared Test for Categorical Features: Statistically assessed
    the association between each categorical feature and ‘Churn’.
    Features like gender and PhoneService were identified as having no
    significant association, while others (e.g., Partner, Contract,
    InternetService) showed strong associations.
-   ANOVA F-test for Numerical Features: Determined the statistical
    association between numerical features and Churn_encoded (binary
    representation of Churn). All numerical features (tenure,
    MonthlyCharges, TotalCharges) showed a highly significant
    association.

4. Data Preprocessing

-   Feature Selection: Selected important features based on EDA and
    statistical tests.
-   One-Hot Encoding: Applied one-hot encoding to categorical features
    to convert them into a numerical format suitable for machine
    learning models.
-   Numerical Feature Transformation:
    -   tenure, MonthlyCharges, TotalCharges were scaled using
        StandardScaler and then transformed using PowerTransformer
        (Yeo-Johnson method) to reduce skewness and achieve a more
        Gaussian-like distribution.
    -   SeniorCitizen was only scaled using StandardScaler as it is a
        binary feature.

5. Model Implementation and Tuning

-   Model: Logistic Regression was chosen as the classification model.
-   Train-Test Split: The dataset was split into training (80%) and
    testing (20%) sets, with stratification to maintain class
    proportions.
-   Initial Model Evaluation: The initial model showed good overall
    accuracy but low recall for the ‘Churn’ class due to class
    imbalance.
-   Threshold Tuning: To address the low recall for churners, the
    classification threshold was tuned from the default 0.5 to 0.3. This
    adjustment prioritizes identifying more actual churners.

Model Performance (After Threshold Tuning to 0.3)

  Metric              Score
  ------------------- --------
  Accuracy            0.7456
  Precision (Churn)   0.5145
  Recall (Churn)      0.7567
  F1-Score (Churn)    0.6126
  ROC AUC             0.8367

Interpretation

The tuned model demonstrates a significant improvement in Recall for the
‘Churn’ class (from ~56% to ~75.7%), meaning it is much more effective
at identifying customers who will churn. This comes with a moderate
decrease in Precision, but the improved F1-Score suggests a better
balance for the minority class, aligning with the business objective of
minimizing missed churners.

Conclusion

This project successfully developed and optimized a Logistic Regression
model for predicting customer churn. By carefully analyzing features,
handling class imbalance, and tuning the classification threshold, the
model can now more effectively identify at-risk customers, providing
valuable insights for proactive retention efforts.

How to Run the Notebook

1.  Clone the repository (if applicable).
2.  Ensure you have the required libraries installed: pandas, numpy,
    matplotlib, seaborn, scikit-learn, scipy, joblib.
3.  Place the Telco.csv dataset in the appropriate directory (e.g.,
    /content/Telco.csv as used in the notebook).
4.  Run all cells in the Jupyter/Colab notebook sequentially.
