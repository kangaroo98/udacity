from config import logging
from churn_library import *


def test_import_data_WrongFileName():
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
        

def test_import_data_MissingColumns():
    try:
        df = import_data("./../data/err_missing_columns.csv")
        logging.error("TEST FAILED: Dataframe returned w/o expection")
        assert False

    except CSV_MissingColumnsError:
        logging.info("TEST SUCCESSFUL: Columns are missing.")
        assert True
    except Exception as err:
        logging.error("TEST FAILED: Wrong Exception: %s", err)
        assert False


def test_import_data_NoRows():
    try:
        df = import_data("./../data/err_no_rows.csv")
        logging.error("TEST FAILED: Dataframe returned w/o expection")
        assert False
    except CSV_NoRowsError:
        logging.info("TEST SUCCESSFUL: No rows in file.")
        assert True
    except Exception as err:
        logging.error("TEST FAILED: Wrong Exception: %s", err)
        assert False
