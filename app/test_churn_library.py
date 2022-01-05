'''
Pytests for the churn library functions

Author: Oliver
Date: 2022 - Jan5
'''
from config import logging
from error import AppError, DfColumnsMismatchError, FileFormatError, FileNoRowsError, EdaError
#from churn_library import import_data
#from churn_library import perform_eda
#from churn_library import encoder_helper
#from churn_library import perform_feature_engineering
from churn_library import *

def test_import_data_filepth():
    '''
    import_data test
    '''
    try:
        df = import_data("blabla")
        logging.error("TEST FAILED: Dataframe returned w/o expection")
        assert False
    except AssertionError as err:
        logging.info("TEST SUCCESSFUL: Filepath does not exist.")
        assert True
    except Exception as err:
        logging.error("TEST FAILED: Wrong Exception: %s", err)
        assert False

def test_import_data_fileformat():
    '''
    import_data test
    '''
    try:
        df = import_data("./../data/err_wrong_file_format.csv")
        logging.error("TEST FAILED: Dataframe returned w/o expection")
        assert False
    except FileFormatError as err:
        logging.info("TEST SUCCESSFUL: Filepath does not exist: %s", err)
        assert True
    except Exception as err:
        logging.error("TEST FAILED: Wrong Exception: %s", err)
        assert False

def test_import_data_missingcols():
    '''
    import_data test
    '''
    try:
        df = import_data("./../data/err_missing_columns.csv")
        logging.error("TEST FAILED: Dataframe returned w/o expection")
        assert False

    except DfColumnsMismatchError:
        logging.info("TEST SUCCESSFUL: Columns are missing.")
        assert True
    except Exception as err:
        logging.error("TEST FAILED: Wrong Exception: %s", err)
        assert False

def test_import_data_no_rows():
    '''
    import_data test
    '''
    try:
        df = import_data("./../data/err_no_rows.csv")
        logging.error("TEST FAILED: Dataframe returned w/o expection")
        assert False
    except FileNoRowsError:
        logging.info("TEST SUCCESSFUL: No rows in file.")
        assert True
    except Exception as err:
        logging.error("TEST FAILED: Wrong Exception: %s", err)
        assert False

def test_perform_eda_imagepth_invalid():
    '''
    perform_eda test
    '''
    try:
        df = import_data("./../data/bank_data.csv")
        perform_eda(df, "blabla")
        logging.error("TEST FAILED: eda performed w/o expection")
        assert False
    except AssertionError:
        logging.info("TEST SUCCESSFUL: pth not available, process failed.")
        assert True
    except Exception as err:
        logging.error("TEST FAILED: Wrong Exception: %s", err)
        assert False

def test_encoder_helper_dfparameter():
    '''
    encoder_helper test
    '''
    try:
        df = import_data("blabla")
        perform_eda(df, "blabla")
        encoder_helper(df, category_columns, "blabla")
        logging.error("TEST FAILED: Enoder returned w/o assertion")
        assert False
    except AssertionError as err:
        logging.info("TEST SUCCESSFUL: Parameter not valid.")
        assert True
    except Exception as err:
        logging.error("TEST FAILED: Wrong Exception: %s", err)
        assert False

def perform_feature_engineering_test():
    pass