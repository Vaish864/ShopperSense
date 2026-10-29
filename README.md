ShopperSense – Online Shopper Purchase Intention Prediction

1. Overview

ShopperSense is a machine learning-based web application that predicts whether an online visitor is likely to make a purchase based on their browsing behaviour. It also provides model comparison, dataset exploration, and customer segmentation features.

The application is developed using Python and Streamlit.

2. Objectives

Predict online shopper purchase intention.

Compare different machine learning classification algorithms.

Explore customer browsing behaviour and dataset patterns.

Identify customer segments using clustering techniques.

Provide an interactive and user-friendly interface.

3. Dataset

The project uses the Online Shoppers Purchasing Intention Dataset from the UCI Machine Learning Repository.

Records: 12,330

Columns: 18

Target variable: Revenue

Source: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

4. Technologies Used

Python – Programming language

Pandas and NumPy – Data processing

Scikit-learn – Machine learning and preprocessing

Joblib – Saving and loading trained models

Matplotlib and Seaborn – Data visualisation

Streamlit – Web application interface

5. Machine Learning Algorithms

The following classification algorithms were implemented and compared:

Random Forest

Decision Tree

Support Vector Machine (SVM)

K-Nearest Neighbours (KNN)

Logistic Regression

K-Means and DBSCAN were also used for customer segmentation and clustering analysis.

6. Key Results

Random Forest achieved the highest F1-score of 68.03%.

Random Forest achieved a ROC-AUC score of 93.09%.

Decision Tree achieved the highest accuracy of 90.21%.

Logistic Regression achieved the highest recall of 78.80%.

These results are based on the project's test-set evaluation.

7. Application Features

Home dashboard with dataset overview.

Model performance comparison.

Interactive purchase intention prediction.

Customer segmentation visualisation.

Dataset exploration and data quality analysis.

8. Project Structure

ShopperSense/
├── app.py
├── requirements.txt
├── online_shoppers_intention.csv
├── shopper_models/
│   ├── random_forest.joblib
│   ├── decision_tree.joblib
│   ├── svm.joblib
│   ├── knn.joblib
│   ├── logistic_regression.joblib
│   ├── kmeans_model.joblib
│   ├── dbscan_model.joblib
│   ├── cluster_preprocessor.joblib
│   └── model_comparison.csv
└── README.md

Note: The folder may contain additional saved models and result files.

9. Installation and Execution

Step 1: Clone the repository

git clone YOUR_GITHUB_REPOSITORY_URL
cd ShopperSense

Replace YOUR_GITHUB_REPOSITORY_URL with your actual GitHub repository URL.

Step 2: Create a virtual environment

python -m venv .venv

Step 3: Activate the environment

Windows PowerShell:

.\.venv\Scripts\Activate.ps1

Step 4: Install dependencies

python -m pip install -r requirements.txt

Step 5: Run the application

streamlit run app.py

The application will open in your web browser.

10. Future Scope

Integrate real-time e-commerce browsing data.

Improve prediction performance through further model tuning.

Add personalised product recommendations.

Develop more detailed customer behaviour analytics.

11. Conclusion

ShopperSense demonstrates how machine learning can be applied to online shopping data to predict purchase intention and analyse customer behaviour. The interactive application combines classification, clustering, model evaluation, and visualisation in a single interface.
