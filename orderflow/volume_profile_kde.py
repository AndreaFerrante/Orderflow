import math
from numba import jit
import numpy as np
from tqdm import tqdm
from numba import prange


def get_vol_dict(p_:np.array, v_:np.array):

    dict_vp = dict()
    all_el = len(p_)

    for i in range(all_el):

        if p_[i] in dict_vp.keys():
            dict_vp[p_[i]] += v_[i]
        else:
            dict_vp[p_[i]] = v_[i]

    return dict_vp


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

    len_source = len(source)
    kde_result = np.zeros(len_source)

    if len_source == 0:
        return kde_result

    jelem, ielem, expo = 0, 0, 0

    ###############################################
    g_const  = 1 / (np.sqrt(2 * np.pi))
    g_const *= 1 / (len_source * h)
    ###############################################

    for j in range(len_source):
        for i in range(len_source):
            expo = (source[j] - source[i]) / h
            ielem = g_const * math.exp(-0.5 * pow(expo, 2))
            if weight[i]:
                ielem *= weight[i]
            jelem += ielem
        kde_result[j] = (jelem)
        jelem = 0

    return kde_result


def gaussian_kde_vectorized(source: np.array, weight: np.array, h: float = 1.0):

    '''
    source : lista dei prezzi
    weight : lista volume
    h : banda (i.e. varianza)
    '''

    len_source = len(source)
    kde_result = np.zeros(len_source)

    if len_source == 0:
        return kde_result

    g_const  = 1 / (np.sqrt(2 * np.pi))
    g_const *= 1 / (len_source * h)

    # creates a matrix of all possible differences between the source elements divided by h.
    # Using np.newaxis creates an additional axis to allow for broadcasting.
    expo = (source[:, np.newaxis] - source) / h
    # applies the exponential to all elements of the matrix
    ielem = g_const * np.exp(-0.5 * np.power(expo, 2))
    # Broadcasting weight array
    # ielem *= weight[:, np.newaxis]
    ielem = np.multiply(ielem, weight)
    # sums the elements along the column axis
    jelem = np.sum(ielem, axis=1)

    kde_result = jelem

    return kde_result


@jit(nopython=True)
def gaussian_kde_numba(source: np.array, weight: np.array, h: float = 1.0):
    len_source = np.shape(source)[0]

    g_const = 1 / (np.sqrt(2 * np.pi))
    g_const *= 1 / (len_source * h)

    expo = (source[:, np.newaxis] - source) / h
    ielem = g_const * np.exp(-0.5 * np.power(expo, 2))
    ielem = np.multiply(ielem, weight)
    jelem = np.sum(ielem, axis=1)

    return jelem


@jit(nopython=True, parallel=True)
def gaussian_kde_numba_parallel(source, weight, h: float = 1.0):
    n = source.shape[0]
    result = np.empty(n, dtype=source.dtype)
    g_const = 1.0 / (np.sqrt(2.0 * np.pi)) / (n * h)

    for j in prange(n):
        sum_val = 0.0
        x_j = source[j]
        for i in range(n):
            expo = (x_j - source[i]) / h
            val = g_const * np.exp(-0.5 * expo * expo)
            if weight.size != 0:
                val *= weight[i]
            sum_val += val
        result[j] = sum_val
    return result


@jit(nopython=True, fastmath=True, parallel=True)
def get_kde_high_low_price_peaks(kde: np.array):

    kde_len = np.shape(kde)[0]
    if kde_len < 2:
        return

    begin_end          = np.array([0, kde_len - 1])
    kde_forward        = np.roll(kde, 1)
    kde_back           = np.roll(kde, -1)
    high_peaks_indexes = np.where((kde > kde_forward) & (kde > kde_back))[0]
    low_peaks_indexes  = np.where((kde < kde_forward) & (kde < kde_back))[0]
    peaks_indexes      = np.concatenate((begin_end, low_peaks_indexes, high_peaks_indexes))

    #################################################
    peaks_indexes = np.unique(np.sort(peaks_indexes))
    #################################################

    return peaks_indexes


