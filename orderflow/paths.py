import os

def get_current_os():

    if os.name == 'posix':
        return ''
    elif os.name == 'nt':
        return r''