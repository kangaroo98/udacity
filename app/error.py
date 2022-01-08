'''
Exception Handling - Definition

Author: Oliver
Date: 2022 - Jan7
'''

# define user-defined exceptions
class AppError(Exception):
    """Base class for other exceptions"""
    pass

class FileFormatError(AppError):
    """Base class for other exceptions"""
    pass

class DfColumnsMismatchError(AppError):
    """Base class for other exceptions"""
    pass

class FileNoRowsError(AppError):
    """Base class for other exceptions"""
    pass

class EdaError(AppError):
    """Base class for other exceptions"""
    pass

class EncodingError(AppError):
    """Base class for other exceptions"""
    pass

class FeatureEngineeringError(AppError):
    """Base class for other exceptions"""
    pass

class ModelTrainingError(AppError):
    """Base class for other exceptions"""
    pass

class ReportingError(AppError):
    """Base class for other exceptions"""
    pass
