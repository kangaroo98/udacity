'''
Pytests for the churn library functions

Author: Oliver
Date: 2022 - Jan5
'''
from config import logging
from error import AppError, FileMissingColumnsError, FileFormatError, FileNoRowsError
from churn_library import import_data


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

    except FileMissingColumnsError:
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
