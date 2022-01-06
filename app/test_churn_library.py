'''
Pytests for the churn library functions
Run command: pytest --capture=no --log-cli-level=INFO test_churn_library.py

Author: Oliver
Date: 2022 - Jan5
'''
import joblib
from config import logging
from config import category_columns, features, target
from app.error import AppError
from app.error import DfColumnsMismatchError, FileFormatError, FileNoRowsError
from churn_library import import_data
from churn_library import perform_eda
from churn_library import encoder_helper
from churn_library import perform_feature_engineering
from churn_library import train_models
from churn_library import classification_report_image
from churn_library import feature_importance_image
from churn_library import compare_roc_image
#from churn_library import *

def test_import_data_filepth():
    '''
    import_data test
    '''
    try:
        print("test")
        import_data("blabla")
        logging.error("TEST FAILED: Dataframe returned w/o expection")
        assert False
    except AssertionError:
        logging.info("TEST SUCCESSFUL: Filepath does not exist.")
        assert True
    except (AppError) as err:
        logging.error("TEST FAILED: Wrong Exception: %s", err)
        assert False


def test_import_data_fileformat():
    '''
    import_data test
    '''
    try:
        import_data("./../data/err_wrong_file_format.csv")
        logging.error("TEST FAILED: Dataframe returned w/o expection")
        assert False
    except FileFormatError as err:
        logging.info("TEST SUCCESSFUL: Filepath does not exist.")
        assert True
    except AppError as err:
        logging.error("TEST FAILED: Wrong Exception: %s", err)
        assert False


def test_import_data_missingcols():
    '''
    import_data test
    '''
    try:
        import_data("./../data/err_missing_columns.csv")
        logging.error("TEST FAILED: Dataframe returned w/o expection")
        assert False

    except DfColumnsMismatchError:
        logging.info("TEST SUCCESSFUL: Columns are missing.")
        assert True
    except AppError as err:
        logging.error("TEST FAILED: Wrong Exception: %s", err)
        assert False


def test_import_data_no_rows():
    '''
    import_data test
    '''
    try:
        import_data("./../data/err_no_rows.csv")
        logging.error("TEST FAILED: Dataframe returned w/o expection")
        assert False
    except FileNoRowsError:
        logging.info("TEST SUCCESSFUL: No rows in file.")
        assert True
    except AppError as err:
        logging.error("TEST FAILED: Wrong Exception: %s", err)
        assert False


def test_perform_eda_imagepth_invalid():
    '''
    perform_eda test
    '''
    try:
        df_eda = import_data("./../data/bank_data.csv")
        perform_eda(df_eda, "blabla")
        logging.error("TEST FAILED: eda performed w/o expection")
        assert False
    except AssertionError:
        logging.info("TEST SUCCESSFUL: pth not available, process failed.")
        assert True
    except AppError as err:
        logging.error("TEST FAILED: Wrong Exception: %s", err)
        assert False


def test_encoder_helper_dfparameter():
    '''
    encoder_helper test
    '''
    try:
        df_encode = import_data("./../data/bank_data.csv")
        perform_eda(df_encode, "./../images/")
        encoder_helper(df_encode, category_columns, "blabla")
        logging.error("TEST FAILED: Enoder returned w/o assertion")
        assert False
    except AssertionError as err:
        logging.info("TEST SUCCESSFUL: Parameter not valid.")
        assert True
    except AppError as err:
        logging.error("TEST FAILED: Wrong Exception: %s", err)
        assert False


def test_perform_feature_engineering_path():
    '''
    perform_feature_engineering test
    '''
    try:
        df_engineering = import_data("./../data/bank_data.csv")
        perform_eda(df_engineering, "./../images/")
        encoder_helper(df_engineering, category_columns, target)
        perform_feature_engineering(df_engineering, features, "blabla")
        logging.error("TEST FAILED: Enoder returned w/o assertion")
        assert False
    except AssertionError as err:
        logging.info("TEST SUCCESSFUL: Parameter not valid.")
        assert True
    except AppError as err:
        logging.error("TEST FAILED: Wrong Exception: %s", err)
        assert False


def test_train_models_feature_target_data_match():
    '''
    train_models test
    '''
    try:
        df_train = import_data("./../data/bank_data.csv")
        perform_eda(df_train, "./../images/")
        encoder_helper(df_train, category_columns, target)
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            df_train, features, target)
        
        # training with incorrect train re test data call
        train_models(x_train, x_test, y_test, y_train)
        logging.error("TEST FAILED: Enoder returned w/o assertion")
        assert False
    except AssertionError as err:
        logging.info("TEST SUCCESSFUL: Parameter not valid.")
        assert True
    except AppError as err:
        logging.error("TEST FAILED: Wrong Exception: %s", err)
        assert False


def test_classification_report_image_path1():
    '''
    classification_report_image test
    '''
    # preparation and analysis
    try:
        df_class = import_data('./../data/bank_data.csv')
        perform_eda(df_class, './../images/')
        encoder_helper(df_class, category_columns, target)

        # feature extraction and model training
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            df_class, features, target)

        # classification report test with incorrect file extension
        classification_report_image(
            joblib.load('./../models/lr_model.pkl'),
            x_train, x_test, y_train, y_test,
            './../images/classification_report_lr.img')

        logging.error("TEST FAILED: classification returned w/o assertion")
        assert False
    except AssertionError as err:
        logging.info("TEST SUCCESSFUL: Parameter not valid.")
        assert True
    except AppError as err:
        logging.error("TEST FAILED: Wrong Exception: %s", err)
        assert False


def test_classification_report_image_path2():
    '''
    classification_report_image test
    '''
    # preparation and analysis
    try:
        df_class = import_data('./../data/bank_data.csv')
        perform_eda(df_class, './../images/')
        encoder_helper(df_class, category_columns, target)

        # feature extraction and model training
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            df_class, features, target)

        # classification report test with missing image dir path
        classification_report_image(
            joblib.load('./../models/lr_model.pkl'),
            x_train, x_test, y_train, y_test,
            'classification_report_lr.png')

        logging.error("TEST FAILED: classification returned w/o assertion")
        assert False
    except AssertionError as err:
        logging.info("TEST SUCCESSFUL: Parameter not valid.")
        assert True
    except AppError as err:
        logging.error("TEST FAILED: Wrong Exception: %s", err)
        assert False


def test_classification_report_image_path3():
    '''
    classification_report_image test
    '''
    try:
        df_class = import_data('./../data/bank_data.csv')
        perform_eda(df_class, './../images/')
        encoder_helper(df_class, category_columns, target)

        # feature extraction and model training
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            df_class, features, target)

        # classification report test with wrong path
        classification_report_image(
            joblib.load('./../models/lr_model.pkl'),
            x_train, x_test, y_train, y_test,
            'blabla')
        
        logging.error("TEST FAILED: classification returned w/o assertion")
        assert False
    except AssertionError as err:
        logging.info("TEST SUCCESSFUL: Parameter not valid.")
        assert True
    except AppError as err:
        logging.error("TEST FAILED: Wrong Exception: %s", err)
        assert False


def test_feature_importance_image_1():
    '''
    feature_importance test
    '''
    try:
        df_imp = import_data('./../data/bank_data.csv')
        perform_eda(df_imp, './../images/')
        encoder_helper(df_imp, category_columns, target)

        # feature extraction and model training
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            df_imp, features, target)

        # image generation test with wrong image file extension
        feature_importance_image(
            joblib.load('./../models/rf_model.pkl'),
            x_test,
            './../images/feature_importance.img')


        logging.error("TEST FAILED: classification returned w/o assertion")
        assert False
    except AssertionError as err:
        logging.info("TEST SUCCESSFUL: Parameter not valid.")
        assert True
    except AppError as err:
        logging.error("TEST FAILED: Wrong Exception: %s", err)
        assert False


def test_compare_roc_image_1():
    '''
    compare_roc_image test
    '''
    try:
        df_roc = import_data('./../data/bank_data.csv')
        perform_eda(df_roc, './../images/')
        encoder_helper(df_roc, category_columns, target)

        # feature extraction and model training
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            df_roc, features, target)

        # image generation test with wrong image file extension
        compare_roc_image(
            joblib.load('./../models/lr_model.pkl'),
            joblib.load('./../models/rf_model.pkl'),
            x_test, y_test,
            './../images/roc_curve_comparison.img')


        logging.error("TEST FAILED: classification returned w/o assertion")
        assert False
    except AssertionError as err:
        logging.info("TEST SUCCESSFUL: Parameter not valid.")
        assert True
    except AppError as err:
        logging.error("TEST FAILED: Wrong Exception: %s", err)
        assert False
