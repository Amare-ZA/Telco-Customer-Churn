Telco Customer Churn Prediction Project

1. Project Overview

This project aims to predict customer churn for a telecommunications
company based on various customer attributes and service usage. The goal is to develop a robust Logistic Regression model that can accurately identify customers at risk of churning, enabling the company to implement targeted retention strategies.

2. Dataset

The dataset, Telco.csv, contains information about a fictional
telecommunications company’s customers, including: Customer
demographics (gender, age, partners, dependents), Services subscribed (phone, multiple lines, internet, online security, backup, device protection, tech support, streaming TV/movies), Account information (contract type, paperless billing, payment method, monthly charges,total charges, tenure), Churn status (Yes/No)

3. Data Preparation and Cleaning

3.1 Handling Missing Values and Duplicates

-   Initial checks confirmed no missing values across most columns, and no duplicate customerIDs.
-   The TotalCharges column, initially of object type, contained some blank spaces, which were treated as NaN and subsequently dropped, resulting in 7032 valid entries.

3.2 Data Type Conversion

-   TotalCharges converted to numeric.
-   Churn encoded as binary variable.

3.3 Multicollinearity

-   Strong correlation (0.83) between tenure and TotalCharges.
-   TotalCharges was dropped to improve model stability.

3.4 Feature Scaling & Transformation

-   One-hot encoding with drop_first=True.
-   StandardScaler + PowerTransformer (Yeo-Johnson) applied to tenure and MonthlyCharges.

4. Exploratory Data Analysis (EDA)

Key churn indicators identified:  Class imbalance (more ‘No’ than
‘Yes’). Senior citizens more likely to churn. No dependents/partners lead to higher churn. Fiber optic users have higher churn. Lack of online services lead to higher churn. Month-to-month contracts lead to the highest churn. PaperlessBilling and Electronic check have higher churn. Low tenure
strongly associated with churn.

5. Feature Selection

Selected features include: Categorical: SeniorCitizen, Partner,
Dependents, MultipleLines, InternetService, OnlineSecurity,
OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
StreamingMovies, Contract, PaperlessBilling, PaymentMethod - Numerical: tenure, MonthlyCharges

Excluded: customerID, gender, PhoneService

6. Model Building and Evaluation

6.1 Initial Logistic Regression

Performance: - Accuracy: 0.8010 - Precision (Churn): 0.6399, Recall
(Churn): 0.5749, F1 (Churn): 0.6056, ROC AUC: 0.8339

6.2 Tuned Logistic Regression 

Performance: Accuracy: 0.7470 - Precision (Churn): 0.5167, Recall
(Churn): 0.7460, F1 (Churn): 0.6105, ROC AUC: 0.8339

7. Conclusion & Next Steps

-   Dropping TotalCharges improved model stability.
-   Balanced class weights significantly improved recall.
-   Accuracy and precision trade-offs are expected in imbalanced
    datasets.
-   Future work: hyperparameter tuning, trying advanced algorithms like
    Gradient Boosting or Random Forest.
