import os

def get_current_os():

    if os.name == 'posix':
        return ''
    elif os.name == 'nt':
        return r'D:\Documenti\Trading\Data\Tick_BidAsk_L2_Data\_SIERRACHART_DATA'