# Stroke Prediction Tool  

## Overview  
This project combines data analysis and machine learning to develop a tool that identifies high-risk individuals experiencing a stroke. The goal is to create a reliable model that can predict the likelihood of stroke based on various health-related features.

## Project Structure  
**1. healthcare-stroke-EDA.ipynb**  
Exploratory Data Analysis (EDA) file.
Statistical summaries, charts, anomaly testing, correlation checks, and other EDA elements.
Statistical inference, including defining the target population, forming hypotheses, constructing confidence intervals, and conducting t-tests.  
**2. modeling.ipynb**  
Modeling process steps.
Data processing using pipelines for cleaning, SMOTE oversampling, feature engineering, and scaling.
Evaluation of 11 baseline models through cross-validation, including Logistic Regression, Random Forest, SVC Linear, SVC RBF, KNN, XGBoost Classifier, LGBM Classifier, Naive Bayes, Ridge Classifier, Linear Discriminant Analysis, and AdaBoost Classifier.
Selection of top three models (AdaBoost Classifier, SVC RBF, and Random Forest) based on F1 scores.
Hyperparameter tuning using Grid Search for fine-tuning the selected models.
Implementation of SHAP (SHapley Additive exPlanations) values for insights and explanations.  
**3. additional2_PYCARET with feat eng**  
Supplementary content file.
Experiment involving automated feature engineering using the PyCaret library to improve model performance.  
**4. Functions**  
.py file containing all classes and functions used in EDA and modeling.  
**5. fastapi**  
Folder containing code for deploying the model on localhost with FastAPI.
How to Contribute
Feel free to contribute to the project by forking the repository and submitting pull requests. Your input is valuable in enhancing the accuracy and efficiency of the stroke prediction tool.  

**Data Source:**
Download the dataset from [here](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).
For any questions or discussions, please connect with @JoneTam.

Thank you for your interest in the Stroke Prediction Project!
