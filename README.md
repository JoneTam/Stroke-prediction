# Stroke-prediction
Goal: This project combines data analysis and machine learning to develop a tool that identifies high-risk individuals experiencing a stroke.

Content:

•	1 healthcare-stroke-EDA.ipynb
Within this file, you'll find exploratory data analysis. This includes creating statistical summaries and charts, testing for anomalies, and checking correlations and other relations between variables and other EDA elements. Also, statistical inference was performed. This includes defining the target population, forming multiple statistical hypotheses and constructing confidence intervals, setting the significance levels, and conducting t-tests for these hypotheses.

•	modeling.ipynb
Contains all the steps of the modeling process.
I employed a series of data processing steps using pipelines, which included data cleaning, SMOTE oversampling, feature engineering, and scaling. These steps were crucial in preparing the final dataset for modeling. The primary success metric for assessing the performance of the stroke prediction models was the F1 score, chosen due to the data's class imbalance.
In the initial phase, I evaluated 11 baseline models through cross-validation, including Logistic Regression, Random Forest, SVC Linear, SVC RBF, KNN, XGBoost Classifier, LGBM Classifier, Naive Bayes, Ridge Classifier, Linear Discriminant Analysis, and AdaBoost Classifier. From this evaluation, I selected the top three models with the highest F1 scores: AdaBoost Classifier, SVC RBF, and Random Forest.
To fine-tune these models, I conducted hyperparameter tuning using Grid Search to identify the optimal set of parameters for each model. Additionally, I employed SHAP (SHapley Additive exPlanations) values to provide insights and explanations for the model's predictions.

•	3 additional2_PYCARET with feat eng
This file includes supplementary content, an experiment involving automated feature engineering. I employed the PyCaret library to investigate whether it could improve the model's performance.

•	Functions
Contains .py file with all classes and functions, used in the EDA and Modeling.

•	fastapi[folder]
Contains code of deploying model on localhost with fast API.
