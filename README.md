# Predict Customer Churn

Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity


## Project Description
Used the Jupyter Notebook "churn_notebook.ipynb" to build a customer churn library in python.
Refactored the notebook code and applied clean code principles. Using autopep8, pylint, pytest.
The project can be accessed in github:


In the root folder of the repo you will find the original "churn_notebook.ipynb". In addition you can use "requirements.txt" to view/install the dependencies.

The project is currently organized in 5 packages/folders:
1. app: source code
- churn_library.py: customer churn library functions
- test_churn_library.py: customer churn test functions (using pytest) 
- config.py: logging configuration and app lib constants 
- error.py: app lib exception definitions
2. logs: customer churn library functions logs, test logging is done directly in the console (see running files below)
- churn_library.log: log file with infos and errors 
3. data: 
- bank_data.csv: original customer data file 
- err_missing_columns.csv: for testing purpose
- err_now_rows.csv: for testing purpose
- err_wrong_file_format.csv: for testing purpose
4. images: reporting image files created by the library functions
5. models: current base are two models - LogisticRegression, Rainforest Classifier

## Running Files
How do you run your files? What should happen when you run your files?

1. I recommend to build a virtual ennvironment.
2. pip install -r requirements.txt
3. git clone

