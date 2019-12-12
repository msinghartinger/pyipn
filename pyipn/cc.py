"""Summary
"""
import numpy as np
from numba import jit


def uccf_ij(xi, mean_x, stdev_x, error_x, yj, mean_y, stdev_y, error_y):
    """Summary

    Args:
        xi (TYPE): Description
        mean_x (TYPE): Description
        stdev_x (TYPE): Description
        error_x (TYPE): Description
        yj (TYPE): Description
        mean_y (TYPE): Description
        stdev_y (TYPE): Description
        error_y (TYPE): Description

    Returns:
        TYPE: Description
    """
    return (xi - mean_x)*(yj - mean_y)/np.sqrt((stdev_x**2 - error_x**2)*(stdev_y**2 - error_y**2))


@jit
def uccf(lc1_x, lc1_y, lc2_x, lc2_y):
    """unbinned cross correlation function

    Args:
        lc1_x (TYPE): Description
        lc1_y (TYPE): Description
        lc2_x (TYPE): Description
        lc2_y (TYPE): Description

    Returns:
        TYPE: Description
    """
    mean1 = np.mean(lc1_y)
    mean2 = np.mean(lc2_y)

    stdev1 = np.std(lc1_y)
    stdev2 = np.std(lc2_y)

    uccf = []
    lags = []

    for i in range(lc1_y.size):
        for j in range(lc2_y.size):
            uccf.append(uccf_ij(lc1_y[i], mean1, stdev1,
                                0., lc2_y[j], mean2, stdev2, 0.))
            lags.append(lc2_x[j]-lc1_x[i])

    uccf = np.array(uccf)
    lags = np.array(lags)
    return uccf, lags


def dcf(uccf, lags, start=-5, stop=5, step=0.1):
    """Summary

    Args:
        uccf (TYPE): Description
        lags (TYPE): Description
        start (TYPE, optional): Description
        stop (int, optional): Description
        step (float, optional): Description

    Returns:
        TYPE: Description
    """
    bins = np.arange(start, stop, step)
    dcf = np.zeros(bins.size)

    for i in range(bins.size):
        lag_pairs = uccf[np.where((lags >= (bins[i]-0.5*step)) &
                                  (lags < (bins[i] + 0.5*step)))]
        dcf[i] = np.mean(lag_pairs)

    return bins, dcf


def lnuccf_ij(xi, lmean_x, lstdev_x, error_x, yj, lmean_y, lstdev_y, error_y):
    """Summary

    Args:
        xi (TYPE): Description
        lmean_x (TYPE): Description
        lstdev_x (TYPE): Description
        error_x (TYPE): Description
        yj (TYPE): Description
        lmean_y (TYPE): Description
        lstdev_y (TYPE): Description
        error_y (TYPE): Description

    Returns:
        TYPE: Description
    """
    return (xi - lmean_x)*(yj - lmean_y)/np.sqrt((lstdev_x**2 - error_x**2)*(lstdev_y**2 - error_y**2))


def lndcf(lc1_x, lc1_y, lc2_x, lc2_y, lags, start=-5, stop=5, step=0.1):
    """Summary

    Args:
        lc1_x (TYPE): Description
        lc1_y (TYPE): Description
        lc2_x (TYPE): Description
        lc2_y (TYPE): Description
        lags (TYPE): Description
        start (TYPE, optional): Description
        stop (int, optional): Description
        step (float, optional): Description

    Returns:
        TYPE: Description
    """

    pairs = []
    lags = []
    for i in range(lc1_y.size):
        for j in range(lc2_y.size):
            pairs.append(np.array([lc1_y[i], lc2_y[j]]))
            lags.append(lc2_x[j]-lc1_x[i])

    pairs = np.array(pairs)
    lags = np.array(lags)

    bins = np.arange(start, stop, step)
    lndcf = np.zeros(bins.size)

    for k in range(bins.size):
        lag_pairs = pairs[np.where((lags >= (bins[k]-0.5*step)) &
                                   (lags < (bins[k] + 0.5*step)))]

        lmean1 = np.mean(lag_pairs[:, 0])
        lmean2 = np.mean(lag_pairs[:, 1])

        lstdev1 = np.std(lag_pairs[:, 0])
        lstdev2 = np.std(lag_pairs[:, 1])

        lnuccf = []
        for p in lag_pairs:
            lnuccf.append(
                lnuccf_ij(p[0], lmean1, lstdev1, 0., p[1], lmean2, lstdev2, 0.))

        lnuccf = np.array(lnuccf)

        lndcf[k] = np.mean(lag_pairs)

    return bins, lndcf
