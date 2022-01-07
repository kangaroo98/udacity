# Predict Customer Churn

Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity


## Project Description
Used the Jupyter Notebook "churn_notebook.ipynb" to build a customer churn library in python.
Refactored the notebook code and applied clean code principles. Using autopep8, pylint, pytest.
The project can be accessed in github:
https://github.com/kangaroo98/udacity.git.

The root folder of the repo contains the original "churn_notebook.ipynb". In addition you can use "requirements.txt" in the root to view/install the dependencies.

The project is currently organized in 5 packages/folders:
1. app: source code
- churn_library.py: customer churn library functions
- test_churn_library.py: customer churn library test functions (using pytest) 
- config.py: logging configuration and app lib constants 
- error.py: app lib exception definitions
2. logs: customer churn library functions logs, test logging is done directly in the console (see running files below)
- churn_library.log: log file with infos and errors 
3. data: 
- bank_data.csv: original customer data file 
- err_missing_columns.csv: for testing purpose
- err_now_rows.csv: for testing purpose
- err_wrong_file_format.csv: for testing purpose
4. images: image reports created by the library functions
- churn_status.png: -> perform_eda
- columns_corr_heatmap.png: -> perform_eda
- marital_status.png: -> perform_eda
- customer_age.png: -> perform_eda
- total_trans_ct.png: -> perform_eda
- classification_report_lr.png: -> classification_report_image
- classification_report_rf.png: -> classification_report_image
- feature_importance.png: -> feature_importance_image
- roc_curve_comparison.png: -> compare_roc_image

5. models: trained models based on the given bank_data.csv
- lr_model.pkl: LogisticRegression
- rf_model.pkl: RandomForestClassifier


## Running Files
How do you run your files? What should happen when you run your files?

1. I recommend to build a virtual ennvironment.
2. git clone https://github.com/kangaroo98/udacity.git
3. cd udacity
4. pip install -r requirements.txt
5. cd app
6. pytest --capture=no --log-cli-level=INFO test_churn_library.py (currently 11 tests should pass, logs will be shown in the console)
7. pytest churn_library.py (__main__ will execute all functions in order, models and images are created and saved, logs can be viewed in ./logs/churn_library.logs)
