"""Summary
"""
import numpy as np
from numba import jit
from scipy.optimize import curve_fit
import scipy as sp

@jit
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
    lc1_y = lc1_y - np.mean(lc1_y)
    lc2_y = lc2_y - np.mean(lc2_y)

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
            lags.append(lc1_x[i]-lc2_x[j])

    uccf = np.array(uccf)
    lags = np.array(lags)
    return uccf, lags


@jit
def dcf(uccf, lags, start=-5, stop=5, step=0.1):
    """
    Create Discrete Cross Correlation Function (DCF; Edelson & Krolik 1988) from unbinned corss correlation function with bin size step
    
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
    sigma_dcf = np.zeros(bins.size)

    for i in range(bins.size):
        lag_pairs = uccf[np.where((lags >= (bins[i]-0.5*step)) &
                                  (lags < (bins[i] + 0.5*step)))]
        dcf[i] = np.mean(lag_pairs)
        sigma_dcf[i] = np.std(lag_pairs)

    return bins, dcf, sigma_dcf


@jit
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


@jit
def lndcf(lc1_x, lc1_y, lc2_x, lc2_y, start=-5, stop=5, step=0.1):
    """
    locally normalized discrete cross correlation function of two (non stationary) time series (light curves)
    
    Args:
        lc1_x (array<float>): x values of light curve 1
        lc1_y (array<float>): y values of light curve 1
        lc2_x (array<float>): x values of light curve 2
        lc2_y (array<float>): y values of light curve 2
        start (TYPE, optional): Description
        stop (int, optional): Description
        step (float, optional): Description
    
    Returns:
        TYPE: Description
    """

    pairs = []
    lags = []

    lc1_y = lc1_y - np.mean(lc1_y)
    lc2_y = lc2_y - np.mean(lc2_y)

    for i in range(lc1_y.size):
        for j in range(lc2_y.size):
            pairs.append(np.array([lc1_y[i], lc2_y[j]]))
            lags.append(lc1_x[i]-lc2_x[j])

    pairs = np.array(pairs)
    lags = np.array(lags)

    bins = np.arange(start, stop, step)
    lndcf = np.zeros(bins.size)
    sigma_lndcf = np.zeros(bins.size)

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
        sigma_lndcf[k] = np.std(lag_pairs)

    return bins, lndcf, sigma_lndcf


@jit
def iccf_ij(lc1_x, lc1_y, lc2_x, lc2_y, tau, method):
    """Summary
    
    Args:
        lc1_x (TYPE): Description
        lc1_y (TYPE): Description
        lc2_x (TYPE): Description
        lc2_y (TYPE): Description
        tau (TYPE): Description
        method (TYPE): Description
    
    Returns:
        TYPE: Description
    """

    ntau = tau.size
    lag_bins = int((ntau-1)/2)

    r_ij = np.zeros(ntau)
    n_ij = lc1_y.size

    sd1 = np.std(lc1_y)
    sd2 = np.std(lc2_y)

    for i in range(ntau):
        if method == 'linear':
            lc2_yinterp = np.interp(lc1_x-tau[i], lc2_x, lc2_y)
        elif method == 'spline':
            spl = sp.interpolate.splrep(lc2_x, lc2_y)
            lc2_yinterp = sp.interpolate.splev(lc1_x-tau[i], spl, ext=3)
        elif method == 'quadratic':
            interp = sp.interpolate.interp1d(lc2_x, lc2_y, kind='quadratic', bounds_error=False, fill_value='extrapolate')
            lc2_yinterp = interp(lc1_x-tau[i])

        r_ij[i] = np.sum(np.multiply(lc1_y, lc2_yinterp))/ sd1 / sd2

    r_ij = r_ij / n_ij
    n_ij = np.zeros(ntau)+n_ij

    return r_ij, n_ij


@jit
def iccf(lc1_x, lc1_y, lc2_x, lc2_y, tau, method='linear'):
    """
    (linearly) Interpolated Cross Correlation Function (ICCF; Gaskell & Sparke 1986)
    transcribed from https://rdrr.io/github/svdataman/sour/src/R/iccf_functions.R
    only method 'linear' works correctly
    
    Args:
        lc1_x (array<float>): x values of light curve 1
        lc1_y (array<float>): y values of light curve 1
        lc2_x (array<float>): x values of light curve 2
        lc2_y (array<float>): y values of light curve 2
        tau (TYPE): Description
        method (str, optional): Description
    
    Returns:
        TYPE: Description
    """
    methods = ["linear", "spline", "quadratic"]
    assert (method in methods), method + "is not a valid method!"

    if method == 'spline':
        start = np.maximum(lc1_x[0], lc2_x[0])
        end = np.minimum(lc1_x[-1], lc2_x[-1])

        lc1_overlap = np.where((lc1_x >= start) and (lc1_x <= end))
        lc1_x = lc1_x[lc1_overlap]
        lc1_y = lc1_y[lc1_overlap]

        lc2_overlap = np.where((lc2_x >= start) and (lc2_x <= end))
        lc2_x = lc2_x[lc2_overlap]
        lc2_y = lc2_y[lc2_overlap]

    n1 = lc1_x.size
    n2 = lc2_x.size

    lc1_y = lc1_y - np.mean(lc1_y)
    lc2_y = lc2_y - np.mean(lc2_y)

    ntau = tau.size
    lag_bins = int(ntau-1)/2

    r_ij, n_ij = iccf_ij(lc1_x, lc1_y, lc2_x, lc2_y, tau, method=method)
    r_ji, n_ji = iccf_ij(lc2_x, lc2_y, lc1_x, lc1_y, -tau, method=method)

    iccf = 0.5*(r_ij+r_ji)

    return tau, iccf

def max_spike_bins(center_bins, center_cf, fit_size, add=0):
    """
    find and return all bins which are close to the main spike in the function
    
    Args:
        center_bins (TYPE): lag values
        center_cf (TYPE): correlation function values
        fit_size (TYPE): include values on both sides of maximum unitl 
                            corr function value lower than fit_size * mean
        add (int, optional): add additional values on each side
    
    Returns:
        TYPE:   lag bins and corresponding corr function values around maximum
                on which to perform fit and lag value of maximum
    """
    argmax = np.argmax(center_cf)
    maxr = center_bins[argmax]
    mean = np.mean(center_cf)

    sup_cent_idx = [argmax]
    rate = mean + 1.
    i = 0
    while rate > (fit_size*mean):
        i += 1
        if not ((argmax+i+1)>center_bins.shape):
            rate = center_cf[argmax+i]
            sup_cent_idx.append(argmax+i)
        else:
            break

    for l in range(add):
        if not ((argmax+i+(l+1)+1)>center_bins.shape):
            sup_cent_idx.append(argmax+i+(l+1))

    rate = mean + 1.
    i = 0
    while rate > (fit_size*mean):
        i -= 1
        if ((argmax+i) >= 0):
            rate = center_cf[argmax+i]
            sup_cent_idx.append(argmax+i)
        else:
            break

    for j in range(add):
        if ((argmax+i-(j+1)) >= 0):
            sup_cent_idx.append(argmax-i-(j+1))

    sup_cent_idx = np.array(sup_cent_idx)
    sup_cent_bins = center_bins[np.where(sup_cent_idx)]
    sup_cent_cf = center_cf[np.where(sup_cent_idx)]

    return sup_cent_bins, sup_cent_cf, maxr


def parabolic_fit(bins, cf, center_size, fit_size):
    """
    parabolic fit of function (x=bins, y=cf) looking only at point that fall within 
    center_size around the maximum with domain fit_size
    
    Args:
        bins (TYPE): Description
        cf (TYPE): Description
        center_size (TYPE): Description
        fit_size (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    center_bins = bins[np.where((bins >= -center_size) & (bins <= center_size))]
    center_cf = cf[np.where((bins >= -center_size) & (bins <= center_size))]
    argmax = center_bins[np.argmax(center_cf)]
    
    sup_cent_bins = center_bins[np.where((center_bins >= argmax-fit_size) & (center_bins <= argmax+fit_size))]
    sup_cent_cf = center_cf[np.where((center_bins >= argmax-fit_size) & (center_bins <= argmax+fit_size))]
    
    parab = np.polyfit(sup_cent_bins, sup_cent_cf, 2)
    maxim = -parab[1]/(2*parab[0])

    return parab, maxim, argmax

def gaussian_fit(bins, cf, center_size, fit_size):
    """
    gaussian fit of function (x=bins, y=cf) looking only at point that fall within 
    center_size around the maximum with domain fit_size
    
    Args:
        bins (TYPE): Description
        cf (TYPE): Description
        center_size (TYPE): Description
        fit_size (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    center_bins = bins[np.where((bins >= -center_size) & (bins <= center_size))]
    center_cf = cf[np.where((bins >= -center_size) & (bins <= center_size))]
    argmax = center_bins[np.argmax(center_cf)]
    
    sup_cent_bins = center_bins[np.where((center_bins >= argmax-fit_size) & (center_bins <= argmax+fit_size))]
    sup_cent_cf = center_cf[np.where((center_bins >= argmax-fit_size) & (center_bins <= argmax+fit_size))]

    def gaus(x, a, x0, sigma, offset):
        return a*np.exp(-(x-x0)**2/(2*sigma**2) + offset)

    popt, pcov = curve_fit(gaus, sup_cent_bins, sup_cent_cf)

    return gaus, popt, argmax


