'''
library to determine customer churn based on two models:
- LogisiticRegression
- RandomForestClassifier
Functions:
- import_data
- perform_eda
- encoder_helper
- perform_feature_engineering
- compare_roc_image
- feature_importance_image
- classification_report_image
- train_models

Author: Oliver
Date: 2022 - Jan7
'''
# import libraries
import os
from sklearn.metrics import classification_report, RocCurveDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from app.error import AppError, DfColumnsMismatchError, FileFormatError, FileNoRowsError
from app.error import EdaError, EncodingError, FeatureEngineeringError
from app.error import ModelTrainingError, ReportingError
from app.config import features, target, param_grid, category_columns, quantitative_columns
from app.config import logging
sns.set()


def import_data(filename):
    '''
    returns dataframe for the csv found at pth
    exceptions:
            FileFormatError: Incorrect file format - cannot read csv
            FileNoRowsError: No rows in the dataset
            DfColumnsMismatchError: predefined columns are missing
                (see config.py - quantitative_columns, category_columns)
    input:
            filename: str - path to the csv
    output:
            df: pandas dataframe
    '''
    # Import the csv data
    logging.info("INFO: Importing file: (%s)", filename)
    assert os.path.isfile(filename)

    try:
        df_csv = pd.read_csv(filename)
    except Exception as err:
        logging.error("ERROR: File could not be read: %s", err)
        raise FileFormatError("ERROR: File could not be read: ",err) from err

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


def perform_eda(df_eda, image_dir_pth):
    '''
    perform eda on df and save figures to images folder.
    created images in image_dir_path:
    churn.png, customer_age.png, marital_status.png, total_trans_ct.png, corr_heatmap.png

    exceptions:
            EdaError: eda re. image(s) creation failed.
    input:
            df_eda: pandas dataframe - initialized by import_data function
            image_pth: str - root images path
    output:
            None
    '''
    logging.info("INFO: performing eda")
    assert os.path.isdir(image_dir_pth)

    try:
        df_eda['Churn'] = df_eda['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)

        plt.figure(figsize=(20, 10))
        df_eda['Churn'].hist()
        plt.savefig(image_dir_pth + "churn.png")
        logging.info("INFO: churn.png image is created in %s", image_dir_pth)

        plt.clf()
        df_eda['Customer_Age'].hist()
        plt.savefig(image_dir_pth + "customer_age.png")
        logging.info(
            "INFO: customer_age.png image is created in %s",
            image_dir_pth)

        plt.clf()
        df_eda.Marital_Status.value_counts('normalize').plot(kind='bar')
        plt.savefig(image_dir_pth + "marital_status.png")
        logging.info(
            "INFO: marital_status.png image is created in %s",
            image_dir_pth)

        plt.clf()
        sns.histplot(df_eda['Total_Trans_Ct'], kde=True, stat="density", linewidth=0)
        plt.savefig(image_dir_pth + "total_trans_ct.png")
        logging.info(
            "INFO: total_trans_ct.png image is created in %s",
            image_dir_pth)

        plt.clf()
        sns.heatmap(df_eda.corr(), annot=False, cmap='Dark2_r', linewidths=2)
        plt.savefig(image_dir_pth + "corr_heatmap.png")
        logging.info(
            "INFO: corr_heatmap.png image is created in %s",
            image_dir_pth)

        logging.info(
            "SUCCESS: all eda images are created in %s",
            image_dir_pth)

    except Exception as err:
        logging.error(
            "ERROR: Visualization failed. Image(s) could not be created:", err)
        raise EdaError(
            "Visualization failed. Image(s) could not be created.") from err
    finally:
        plt.close()


def encoder_helper(df_encode, category_lst, rel_column):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook
    exceptions:
            EncodingError: processing error
    input:
            df_encode: pandas dataframe - \
                initialized by import_data function followed by perform_eda function
            category_lst: str list of column names that contain categorical features \
                (must be part of df_encode)
            rel_column: str - related column
    output:
            df_encode: pandas dataframe with new columns
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
    prepare / split train and test data

    exceptions:
            FeatureEngineeringError: processing error
    input:
            df_org: pandas dataframe
            feature_name_list: features by name
            target_column_name: target by name
    output: train_test_split results:
              X training data
              X testing data
              y training data
              y testing data
    '''
    logging.info(
        "INFO: Preparing train and test data - features: %s and target: %s",
        feature_name_list,
        target_column_name)
    assert set(feature_name_list +
               [target_column_name]).issubset(set(df_org.columns))

    try:
        # feature/target extraction
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
        raise FeatureEngineeringError("Feature engineering failed.") from err


def compare_roc_image(
        lr_model,
        rf_model,
        features_test,
        target_test,
        image_file_pth):
    '''
    plotting a comparison of lr and rf and save image in image_file_pth

    exceptions:
        ReportingError
    input:
        lr_model: logistic regression model
        rf_model: rain forest model
        features_test: feature data
        target_test: target data
        image_file_path: str
    output:
        None
    '''
    logging.info("INFO: roc comparison image generation")
    image_dir, image_name = os.path.split(image_file_pth)
    assert os.path.isdir(image_dir)
    assert image_name[-4:] == '.png'

    try:
        plt.figure(figsize=(20, 10))
        axes = plt.gca()
        RocCurveDisplay.from_estimator(
            lr_model,
            features_test,
            target_test,
            ax=axes,
            alpha=0.8)
        RocCurveDisplay.from_estimator(
             rf_model,
             features_test,
             target_test,
             ax=axes,
             alpha=0.8)
        plt.savefig(image_file_pth)
        logging.info("SUCCESS: roc image saved as %s", image_file_pth)

    except Exception as err:
        logging.error("ERROR: Comparison roc failed:", err)
        raise ReportingError("Comparison roc failed.") from err
    finally:
        plt.close()


def feature_importance_image(rf_model, feature_data, image_file_pth):
    '''
    creates and stores the feature importance plot as image_file_pth

    exceptions:
            ReportingError
    input:
            rf_model: rf model object containing feature_importances_
            feature_data: pandas dataframe of feature values
            output_pth: path to store the figure
    output:
             None
    '''
    logging.info("INFO: feature importance plot generation")
    image_dir, image_name = os.path.split(image_file_pth)
    assert os.path.isdir(image_dir)
    assert image_name[-4:] == '.png'

    try:

        # Calculate feature importances
        importances = rf_model.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [feature_data.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20, 20))

        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')

        # Add bars
        plt.bar(range(feature_data.shape[1]), importances[indices])

        # Add feature names as x-axis labels
        plt.xticks(range(feature_data.shape[1]), names, rotation=90)

        plt.savefig(image_file_pth)
        logging.info("SUCCESS: image saved as %s", image_file_pth)

    except Exception as err:
        logging.error("ERROR: Feature importance plot failed:", err)
        raise ReportingError("Feature importance plot failed.") from err
    finally:
        plt.close()


def classification_report_image(
        model,
        train_features,
        test_features,
        train_target,
        test_target,
        image_file_pth):
    '''
    produces classification report for training and testing results
    and stores report as image_file_path

    exceptions:
        ReportingError
    input:
        model
        train_features: features train data
        test_features: features test data
        train_target: target train data
        test_target: target test data
        image_file_pth: str - filepath of the saved image
    output:
        None
    '''
    logging.info("INFO: classification report generation")
    image_dir, image_name = os.path.split(image_file_pth)
    assert os.path.isdir(image_dir)
    assert image_name[-4:] == '.png'

    try:
        preds_train = model.predict(train_features)
        preds_test = model.predict(test_features)

        plt.figure(figsize=(10, 10))
        plt.rc('figure', figsize=(5, 5))
        plt.text(0.01, 1, str(image_name[:-4] + " Train"),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.7, str(
                classification_report(
                    train_target, preds_train)), {
                'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.6, str(image_name[:-4] + " Test"),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.text(
            0.01, 0.3, str(
                classification_report(
                    test_target, preds_test)), {
                'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')

        plt.savefig(image_file_pth)
        logging.info("SUCCESS: classification report image saved as %s", image_file_pth)

    except Exception as err:
        logging.error(
            "ERROR: Classification report image generation failed:", err)
        raise ReportingError(
            "Classification report image generation failed.") from err
    finally:
        plt.close()


def train_models(train_features, test_features, train_target, test_target):
    '''
    train, store model results: images + scores, and store models
    exception:
            ModelTrainingError
    input:
            train_features: X training data
            test_features: X testing data
            train_target: y training data
            test_target: y testing data
    output:
            models: lrc, rfc
    '''
    logging.info("INFO: training models")
    assert train_features.shape[0] > 1
    assert test_features.shape[0] > 1
    assert train_features.shape[0] == train_target.shape[0]
    assert test_features.shape[0] == test_target.shape[0]

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

        logging.info('INFO: logistic regression results')
        logging.info('INFO: test results')
        logging.info(classification_report(test_target, test_preds_lr))
        logging.info('INFO: train results')
        logging.info(classification_report(train_target, train_preds_lr))

        return lrc, cv_rfc.best_estimator_

    except Exception as err:
        logging.error("ERROR: Training failed:", err)
        raise ModelTrainingError("Training failed.") from err


if __name__ == "__main__":

    try:
        # preparation and analysis
        df = import_data('./../data/bank_data.csv')
        perform_eda(df, './../images/')
        encoder_helper(df, category_columns, target)

        # feature extraction and model training
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            df, features, target)
        # lr, rf = train_models(X_train, X_test, y_train, y_test)
        # joblib.dump(lr, './../models/lr_model.pkl')
        # joblib.dump(rf, './../models/rf_model.pkl')

        # generate reports / images
        classification_report_image(
            joblib.load('./../models/lr_model.pkl'),
            X_train, X_test, y_train, y_test,
            './../images/classification_report_lr.png')
        classification_report_image(
            joblib.load('./../models/rf_model.pkl'),
            X_train, X_test, y_train, y_test,
            './../images/classification_report_rf.png')
        feature_importance_image(
            joblib.load('./../models/rf_model.pkl'),
            X_test,
            './../images/feature_importance.png')
        compare_roc_image(
            joblib.load('./../models/lr_model.pkl'),
            joblib.load('./../models/rf_model.pkl'),
            X_test, y_test,
            './../images/roc_curve_comparison.png')

    except (Exception) as error:
        print("Library error: %s", error)
