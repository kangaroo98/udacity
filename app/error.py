# define user-defined exceptions
class AppError(Exception):
    """Base class for other exceptions"""
    pass

class CSV_WrongFileTypeError(AppError):
    """Base class for other exceptions"""
    pass

class CSV_MissingColumnsError(AppError):
    """Base class for other exceptions"""
    pass

class CSV_NoRowsError(AppError):
    """Base class for other exceptions"""
    pass