"""
Exceptions thrown by orderflow package that are specific to this package only
"""


class SessionTypeAbsent(BaseException):
    """Raised when the column SessionType is not present inside DataFrame"""
    pass