'''
Exception Handling - Definition

Author: Oliver
Date: 2022 - Jan7
'''

# define user-defined exceptions
class AppError(Exception):
    ''' base class for app exceptions '''

class FileFormatError(AppError):
    ''' csv file format exception '''

class DfColumnsMismatchError(AppError):
    ''' dataframe does not contain mandatory columns exception '''

class FileNoRowsError(AppError):
    ''' Base class for other exception '''

class EdaError(AppError):
    ''' perform eda exception '''

class EncodingError(AppError):
    ''' data frame encoding exception '''

class FeatureEngineeringError(AppError):
    ''' train/test data engineering exception '''

class ModelTrainingError(AppError):
    ''' model training exception '''

class ReportingError(AppError):
    ''' reporting excdeption '''
