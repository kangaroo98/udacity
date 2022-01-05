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
