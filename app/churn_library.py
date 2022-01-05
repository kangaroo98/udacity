'''
library to determine customer churn based on two models:
- LogisiticRegression
- RandomForestClassifier

Functions:
- import_data(pth)
- perform_eda(df_eda, pth)
- encoder_helper(df_encode, category_lst, rel_column)
- perform_feature_engineering(df_org, feature_name_list, target_column_name)
- compare_lr_rf_model(
        target_test,
        features_test,
        rfc_model_name,
        lr_model_name)
- feature_importance_plot(model, feature_data, output_pth)
- classification_report_image(model_name,
                                target_train,
                                target_test,
                                train_preds,
                                test_preds):
- train_models(train_features, test_features, train_target, test_target)

Author: Oliver
Date: 2022 - Jan5
'''
# import libraries
import os
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import normalize
#import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from app.error import AppError, DfColumnsMismatchError
from app.error import FileFormatError, FileNoRowsError, EdaError, EncodingError
from app.config import features, target, param_grid, category_columns, quantitative_columns
from app.config import logging
sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth
    exceptions:
            FileFormatError: Incorrect file format - cannot read csv
            FileNoRowsError: No rows in the dataset
            FileMissingColumnsError: predefined columns are missing
                (see config.py - quantitative_columns, category_columns)
    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    # Import the csv data
    logging.info("INFO: Importing file: (%s)", pth)
    assert os.path.exists(pth)

    try:
        df_csv = pd.read_csv(pth)
    except Exception as err:
        logging.error("ERROR: File could not be read: %s", err)
        raise FileFormatError("ERROR: File could not be read: %s", 
            err) from err

    # Validate the imported data for further processing
    if df_csv.shape[0] <= 1:
        logging.info("INFO: Test data imported. Rows: %s ",
                     df_csv.shape[0])
        logging.error("ERROR: File contains no data to train.")
        raise FileNoRowsError("File contains no training data")
    if not set(quantitative_columns +
               category_columns).issubset(set(df_csv.columns)):
        logging.info("INFO: Imported column names: %s", df_csv.columns)
        logging.info(
            "INFO: Column names expected: %s",
            quantitative_columns +
            category_columns)
        logging.error(
            "ERROR: Imported columns do NOT match. Further processing not possible.")
        raise DfColumnsMismatchError(
            "File corrupted. Missing Columns. Please refer to mandatory columns \
                defined in config.py (quantitative_columns, category_columns)")

    logging.info(
        "SUCCESS: File imported, dataframe created containing %s rows.",
        df_csv.shape[0])
    return df_csv


def perform_eda(df_eda, pth):
    '''
    perform eda on df and save figures to images folder

    exceptions:
            EdaError: eda re. image(s) creation failed.
    input:
            df_eda: pandas dataframe - initialized by import_data function
            pth: root images path
    output:
            None
    '''
    logging.info("INFO: perfroming eda: (%s)", pth)
    assert os.path.exists(pth)

    try:
        df_eda['Churn'] = df_eda['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)

        plt.figure(figsize=(20, 10))
        df_eda['Churn'].hist()
        plt.savefig(pth + "churn.png")

        plt.figure(figsize=(20, 10))
        df_eda['Customer_Age'].hist()
        plt.savefig(pth + "customer_age.png")

        plt.figure(figsize=(20, 10))
        df_eda.Marital_Status.value_counts('normalize').plot(kind='bar')
        plt.savefig(pth + "status.png")

        plt.figure(figsize=(20, 10))
        sns.distplot(df_eda['Total_Trans_Ct'])
        plt.savefig(pth + "total_trans_ct.png")

        plt.figure(figsize=(20, 10))
        sns.heatmap(df_eda.corr(), annot=False, cmap='Dark2_r', linewidths=2)
        plt.savefig(pth + "heatmap.png")

        logging.info("SUCCESS: Images are created in %s.", pth)

    except Exception as err:
        logging.error(
            "ERROR: Visualization failed. Image(s) could not be created:", err)
        raise EdaError(
            "Visualization failed. Image(s) could not be created.") from err


def encoder_helper(df_encode, category_lst, rel_column):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook
    exceptions:
            EncodingError: processing error
    input:
            df_encode: pandas dataframe - \
                initialized by import_data function followed by perform_eda function
            category_lst: list of columns that contain categorical features \
                (must be part of df_encode)
            rel_column: related column
    output:
            df_encode: pandas dataframe with new columns for
    '''
    logging.info(
        "INFO: encoding category list: %s with related column: %s",
        category_lst,
        rel_column)
    assert set(category_lst + [rel_column]).issubset(set(df_encode.columns))

    try:
        # encoding category columns
        for category in category_lst:

            lst = []
            groups = df_encode.groupby(category).mean()[rel_column]

            for val in df_encode[category]:
                lst.append(groups.loc[val])

            new_col_name = str(category + '_' + rel_column)
            df_encode[new_col_name] = lst
            logging.info(
                "SUCCESS: Added new category encoded column (%s)",
                new_col_name)

    except Exception as err:
        logging.error("ERROR: Encoding failed:", err)
        raise EncodingError("Encoding the categories failed.") from err


def perform_feature_engineering(df_org, feature_name_list, target_column_name):
    '''
    input:
              df_org: pandas dataframe
              feature_name_list: all features by name
              target_column_name: related column
              [optional argument that could be used for naming variables or index y column]
    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    try:
        # feature extraction
        col_target = df_org[target_column_name]
        df_features = pd.DataFrame()
        df_features[feature_name_list] = df_org[feature_name_list]

        # train and test data split
        return train_test_split(
            df_features,
            col_target,
            test_size=0.3,
            random_state=42)

    except Exception as err:
        logging.error("ERROR: Feature engineering failed:", err)
        raise AppError("Feature engineering failed.") from err


def compare_lr_rf_model(
        target_test,
        features_test,
        rfc_model_name,
        lr_model_name):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            taregt_tet: training response values
            features_test:  test response values
            rfc_model_name: str
            lr_model_name: str
    output:
             None
    '''
    try:
        rfc_model = joblib.load(rfc_model_name)
        lr_model = joblib.load(lr_model_name)

        lrc_plot = plot_roc_curve(lr_model, features_test, target_test)
        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        rfc_disp = plot_roc_curve(
            rfc_model,
            features_test,
            target_test,
            ax=ax,
            alpha=0.8)
        lrc_plot.plot(ax=ax, alpha=0.8)
        plt.savefig("./../images/classification_report_comparison.png")

    except Exception as err:
        logging.error("ERROR: Comparison lr vs rf model failed:", err)
        raise AppError("Comparison lr vs rf model failed.") from err


def feature_importance_plot(model, feature_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            feature_data: pandas dataframe of feature values
            output_pth: path to store the figure
    output:
             None
    '''
    try:

        # Calculate feature importances
        importances = model.best_estimator_.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [feature_data.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20, 5))

        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')

        # Add bars
        plt.bar(range(feature_data.shape[1]), importances[indices])

        # Add feature names as x-axis labels
        plt.xticks(range(feature_data.shape[1]), names, rotation=90)

        plt.savefig(output_pth)

    except Exception as err:
        logging.error("ERROR: Feature importance plot failed:", err)
        raise AppError("Feature importance plot failed.") from err


def classification_report_image(model_name,
                                target_train,
                                target_test,
                                train_preds,
                                test_preds):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest
    output:
             None
    '''
    try:
        plt.figure(figsize=(20, 5))
        plt.rc('figure', figsize=(5, 5))
        plt.text(0.01, 1.25, str(model_name + "_Test"),
                 {'fontsize': 10}, fontproperties='monospace')
        # approach improved by OP -> monospace!
        plt.text(
            0.01, 0.05, str(
                classification_report(
                    target_test, test_preds)), {
                'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.6, str(model_name + "_Train"),
                 {'fontsize': 10}, fontproperties='monospace')
        # approach improved by OP -> monospace!
        plt.text(
            0.01, 0.7, str(
                classification_report(
                    target_train, train_preds)), {
                'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')

        plt.savefig(
            str("./../images/classification_report_" + model_name + ".png"))

    except Exception as err:
        logging.error(
            "ERROR: Classification report image generation failed:", err)
        raise AppError(
            "Classification report image generation failed.") from err


def train_models(train_features, test_features, train_target, test_target):
    '''
    train, store model results: images + scores, and store models
    input:
              train_features: X training data
              test_features: X testing data
              train_target: y training data
              test_target: y testing data
    output:
              None
    '''
    try:
        # grid search
        rfc = RandomForestClassifier(random_state=42)
        lrc = LogisticRegression()

        # train models
        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(train_features, train_target)
        lrc.fit(train_features, train_target)

        train_preds_lr = lrc.predict(train_features)
        train_preds_rf = cv_rfc.best_estimator_.predict(train_features)
        test_preds_lr = lrc.predict(test_features)
        test_preds_rf = cv_rfc.best_estimator_.predict(test_features)

        # classification reports
        logging.info('INFO: random forest results')
        logging.info('INFO: test results')
        logging.info(classification_report(test_target, test_preds_rf))
        logging.info('INFO: train results')
        logging.info(classification_report(train_target, train_preds_rf))

        classification_report_image(
            'Random Forest',
            train_target,
            test_target,
            train_preds_rf,
            test_preds_rf)

        logging.info('INFO: logistic regression results')
        logging.info('INFO: test results')
        logging.info(classification_report(test_target, test_preds_lr))
        logging.info('INFO: train results')
        logging.info(classification_report(train_target, train_preds_lr))

        classification_report_image(
            'Logistic Regression',
            train_target,
            test_target,
            train_preds_lr,
            test_preds_lr)

        feature_importance_plot(
            cv_rfc,
            test_features,
            "./../images/feature_importance.png")

        # save best model
        joblib.dump(cv_rfc.best_estimator_, './../models/rfc_model.pkl')
        joblib.dump(lrc, './../models/logistic_model.pkl')

    except Exception as err:
        logging.error("ERROR: Training failed:", err)
        raise AppError("Training failed.") from err


if __name__ == "__main__":

    try:
        # preparation and analysis
        df = import_data("./../data/bank_data.csv")
        perform_eda(df, "./../images/")
        encoder_helper(df, category_columns, target)

        # feature extraction and model training
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            df, features, target)
        # train_models(X_train, X_test, y_train, y_test)

        # compare_lr_rf_model(
        #     y_test,
        #     X_test,
        #     './../models/rfc_model.pkl',
        #     './../models/logistic_model.pkl')

    except Exception as e:
        print("Error!!! Detail:", e)
