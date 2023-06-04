from collections import deque
import numpy as np
from tqdm import tqdm


def gaussian_kde(source:np.array, weight:np.array, h:float=1.0):

    '''
    Volume profile is just a histogram which means that certain bars could not correctly indicate valleys while
    the biggest bars surely indicates peaks. This is not good for analysis since there is no smooth.
    To smooth out and have a continuous curve we use Kernel Density Estimation. We could use moving averages, but they
    have lags which could lag decisions entries, too.

    source: this is the price (level) volume profile is pointing to
    weight: this is the volume profile weight for each single price level (a groupby price with sum of volume)
    h: standard deviation (i.e. the "length") of the Gaussian Kernel (e.g. 1 for white noise)
    '''

    kde_result = deque()

    if len(source) == 0:
        return kde_result

    len_source = len(source)
    len_weight = len(weight)
    jelem, ielem, expo = 0, 0, 0

    ###############################################
    g_const  = 1 / (np.sqrt(2 * np.pi))
    g_const *= 1 / (len_source * h)
    ###############################################

    print(f'\nPerforming Gaussian KDE, please wait.\n')

    for j in tqdm(range(len_source)):
        for i in range(len_weight):
            expo  = (source[j] - source[i]) / h
            ielem = g_const * np.exp(-0.5 * (expo ** 2))
            if weight[i]:
                ielem *= weight[i]
            jelem += ielem
        kde_result.append(jelem)
        jelem = 0

    return kde_result



def get_peak_valleys():

    '''
    A peak is where both on the left and on the right of the KDE curve is smaller than the peak itself.
    A valley is where both on the left and on the right of the KDE curve is bigger than the peak itself.
    '''

    pass



