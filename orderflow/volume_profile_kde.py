from collections import deque
import numpy as np

def gaussian_kde(source, weight, h):

    '''
    Volume profile is just a histogram which means that certain bars could not correctly indicate valleys while
    the biggest bars surely indicates peaks. This is not good for analysis since there is no smooth.
    To smooth out and have a continuous curve we use Kernel Density Estimation. We could use moving averages, but they
    have lags which could lag decisions entries, too.

    source: this is the price (level) volume profile is pointing to
    weight: this is the volume
    h: standard deviation of the Gaussian Kernel (e.g. 1 for white noise)
    '''

    kde_result = deque()



def get_peak_valleys():

    '''
    A peak is where both on the left and on the right of the KDE curve is smaller than the peak itself.
    A valley is where both on the left and on the right of the KDE curve is bigger than the peak itself.
    '''

    pass



