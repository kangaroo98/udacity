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
After having problems with the provided udacity workspace stability, I decided to develop it locally and share the code in github. My local python version is 3.9.3., but it runs also in your environment based on python 3.6.3. (although some depr.code warnings appear with pytest). Please be aware that currently a relative path to the project root is used throughout the application (inc. loading/saving logs, data, images, models). 
1. git clone https://github.com/kangaroo98/udacity.git
2. cd udacity
3. either install missing libraries via the udacity env. or upgrade to 3.9.3 and pip install -r requirements.txt 
4. python ./app/churn_library.py - (__main__) will execute all functions in order, models and images are created and saved, logs can be viewed in ./logs/churn_library.logs
5. python -m pytest --capture=no --log-cli-level=INFO ./app/test_churn_library.py - logs will be shown in the console
6. pylint result for churn_library.py and test_churn_library.py > 8
