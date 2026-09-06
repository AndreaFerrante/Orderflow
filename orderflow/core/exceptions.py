"""
Exceptions thrown by orderflow package that are specific to this package only
"""


class SessionTypeAbsent(BaseException):
    """Raised when the column SessionType is not present inside DataFrame"""
    pass

class DatetimeTypeAbsent(BaseException):
    """Raised when the column Datetime is not present inside DataFrame"""
    pass

class IndexAbsent(BaseException):
    """Raised when the column Index is not present inside DataFrame"""
    pass

class ColumnNotPresent(BaseException):
    """Raised when a column searched is not present in a dataframe"""
    pass