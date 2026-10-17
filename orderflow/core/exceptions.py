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


def _call_kernel(kernel, *arguments):
    """
    ``kernel(*arguments)`` for a Numba-compiled function, with Ctrl-C still a ``KeyboardInterrupt``.

    Numba calls a Python helper while it hands arrays back. An interrupt pending by then is raised
    inside that helper and leaves the call as ``SystemError: ... returned a result with an exception
    set``: ``except KeyboardInterrupt`` misses it and ``except Exception`` swallows it. The Python
    loops the kernels stand in for raised ``KeyboardInterrupt``.
    """
    try:
        return kernel(*arguments)
    except SystemError as error:
        if isinstance(error.__cause__, KeyboardInterrupt):
            raise error.__cause__ from None
        raise
