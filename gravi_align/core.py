from bisect import bisect_left, insort
from collections import deque
from itertools import islice

import numpy as np
import pkg_resources
from astropy.io import fits
from astropy.modeling import fitting, models
from matplotlib import pyplot as plt
from scipy.ndimage import shift
from scipy.signal import correlate
from termcolor import cprint

file_tellu = pkg_resources.resource_stream(
    "gravi_align", "internal_data/Telluric_lines.txt"
)

tellu = np.loadtxt(file_tellu, skiprows=1)


def open_spectrum_file(file, nint=25):
    file_raw = file.split("aligned.fits")[0] + ".fits"

    try:
        data = fits.getdata(file_raw, "SPECTRUM_DATA_SC")
        flux = np.array([data["DATA%i" % i].sum(axis=0) for i in np.arange(1, nint)])
    except FileNotFoundError:
        flux = None

    data = fits.getdata(file, "SPECTRUM_DATA_SC")
    flux_align = np.array([data["DATA%i" % i].sum(axis=0) for i in np.arange(1, nint)])
    wave_align = fits.open(file)["OI_WAVELENGTH", 10].data.field("EFF_WAVE") * 1e6

    return flux, wave_align, flux_align


def _running_median(seq, M):
    """
     Purpose: Find the median for the points in a sliding window (odd number in size)
              as it is moved from left to right by one point at a time.
      Inputs:
            seq -- list containing items for which a running median (in a sliding window)
                   is to be calculated
              M -- number of items in window (window size) -- must be an integer > 1
      Otputs:
         medians -- list of medians with size N - M + 1
       Note:
         1. The median of a finite list of numbers is the "center" value when this list
            is sorted in ascending order.
         2. If M is an even number the two elements in the window that
            are close to the center are averaged to give the median (this
            is not by definition)
    """
    seq = iter(seq)
    s = []
    m = M // 2

    # Set up list s (to be sorted) and load deque with first window of seq
    s = [item for item in islice(seq, M)]
    d = deque(s)

    # Simple lambda function to handle even/odd window sizes
    def median():
        return s[m] if bool(M & 1) else (s[m - 1] + s[m]) * 0.5

    # Sort it in increasing order and extract the median ("center" of the sorted window)
    s.sort()
    medians = [median()]

    # Now slide the window by one point to the right for each new position (each pass through
    # the loop). Stop when the item in the right end of the deque contains the last item in seq
    for item in seq:
        old = d.popleft()  # pop oldest from left
        d.append(item)  # push newest in from right
        # locate insertion point and then remove old
        del s[bisect_left(s, old)]
        # insert newest such that new sort is not required
        insort(s, item)
        medians.append(median())
    return medians


def _substract_run_med(spectrum, wave=None, n_box=50, shift_wl=0, div=False):
    """ Substract running median from a raw spectrum `f`. The median
    is computed at each points from n_box/2 to -n_box/2+1 in a
    'box' of size `n_box`. The Br gamma line in vaccum and telluric
    lines can be displayed if wavelengths table (`wave`) is specified.
    `shift_wl` can be used to shift wave table to estimate the
    spectral shift w.r.t. the telluric lines.
    """
    # Reference spectral lines
    brg = 2.16612

    r_med = _running_median(spectrum, n_box)
    boxed_flux = spectrum[n_box // 2 : -n_box // 2 + 1]

    boxed_wave = np.arange(len(boxed_flux))
    if wave is not None:
        boxed_wave = wave[n_box // 2 : -n_box // 2 + 1] - shift_wl

    fontsize = 12
    if False:
        plt.figure(figsize=[13, 8])
        ax1 = plt.subplot(211)
        plt.plot(
            boxed_wave,
            boxed_flux,
            label=r"Raw spectrum ($\lambda_{off}$ = %2.2fnm)" % (shift_wl * 1000),
        )
        plt.plot(boxed_wave, r_med, label="Running median")
        plt.axvline(brg, color="#008f53", label=r"Br$\gamma$ line")
        plt.legend(fontsize=fontsize)
        plt.ylabel("Flux [counts]", fontsize=fontsize)
        plt.xlabel(r"Wavelengths [$\mu$m]", fontsize=fontsize)
        plt.subplot(212, sharex=ax1)
        plt.plot(boxed_wave, boxed_flux - r_med)
        plt.ylabel("Normalized flux [counts]", fontsize=fontsize)
        plt.xlabel(r"Wavelengths [$\mu$m]", fontsize=fontsize)
        plt.axvline(brg, color="#008f53", label=r"Br$\gamma$ line")
        plt.tight_layout()

    res = boxed_flux - r_med
    if div:
        res = boxed_flux / r_med

    return res, boxed_wave


def compute_corr_map(
    l_spec,
    wave,
    ref_index=0,
    n_box=50,
    mean=True,
    master_ref=None,
    brg=[2.165, 2.168],
    corr_lim=[2.13, 2.2],
    div=False,
):
    """ Compute the 2D correlation map of several spectra
    using the spectrum number `ref_index` as reference.

    Parameters:
    -----------
    `list_spectrum` {array}:
        List of spectrum (e.g.: 24 for GRAVITY),\n
    `ref_index` {int}:
        Index of the reference spectrum (if `mean`=False),\n
    `n_box` {int}:
        Size of the box to compute the running median,\n
    `mean` {bool}:
        If True, the averaged spectrum is used as reference,\n
    `brg` {list}:
        Br$\gamma$ position [$\mu$m] to be excluded from the correlation,\n
    `corr_lim` {list}:
        Range in $\mu$m to perform the correlation (default is around
        Br$\gamma$, i.e.: [2.13, 2.2]),\n
    `display` {bool}:
        If True, plot the reference spectrum normalized.

    Outputs:
    --------
    `corr_map` {array}:
        List of correlation between spectrum,\n
    `boxed_wave` {array}:
        Resized wave table after normalization.
    """
    # Reference spectrum to compute the correlation matrix

    n_spec = len(l_spec)
    if mean:
        l_norm_spec = []
        for i in range(n_spec):
            f_tmp, boxed_wave = _substract_run_med(
                l_spec[i], wave=wave, n_box=n_box, div=div
            )
            l_norm_spec.append(f_tmp)
        l_norm_spec = np.array(l_norm_spec)

        ref_spectrum = np.mean(l_norm_spec, axis=0)
    else:
        ref_spectrum, boxed_wave = _substract_run_med(
            l_spec[ref_index], wave=wave, n_box=n_box, div=div
        )

    if master_ref is not None:
        ref_spectrum = master_ref

    n_spec = l_spec.shape[0]
    cond_BrG = (boxed_wave >= brg[0]) & (boxed_wave <= brg[1])
    cond_range = (boxed_wave >= corr_lim[0]) & (boxed_wave <= corr_lim[1])
    cond_sel = cond_range & ~cond_BrG

    ref_spectrum_sel = ref_spectrum[cond_sel]
    size_spectr_norm = ref_spectrum_sel.shape[0]
    n_corr = (size_spectr_norm * 2) - 1

    corr_map = np.zeros([n_spec, n_corr])
    for i in range(n_spec):
        inp_spectre = l_spec[i]
        spec_to_compare = _substract_run_med(inp_spectre, n_box=n_box)[0]
        corr_tmp = correlate(ref_spectrum_sel, spec_to_compare[cond_sel])
        corr_map[i] = corr_tmp
    return corr_map


def compute_shift(corr_map, size=5):
    """ Fit the correlation peaks with a Gaussian model on all
    the 24 spectrum outputs of GRAVITY.
    """
    fitter_gauss = fitting.LevMarLSQFitter()

    n_spec = corr_map.shape[0]
    n_channel = corr_map.shape[1]
    pos_wl0 = n_channel // 2
    x_corr = np.arange(n_channel) - pos_wl0

    l_shift_corr = np.zeros(n_spec)
    for i_fit in range(n_spec):
        cond_sel = (x_corr < size) & (x_corr > -size)
        x_to_fit = x_corr[cond_sel]
        y_to_fit = corr_map[i_fit][cond_sel]
        g_init = models.Gaussian1D(amplitude=y_to_fit.max(), mean=1, stddev=1.0)
        g = fitter_gauss(g_init, x_to_fit, y_to_fit)
        l_shift_corr[i_fit] = g.mean.value

    x_fitted = pos_wl0 + l_shift_corr
    label_fitted = ["%2.3f" % x for x in l_shift_corr]
    y_spectrum = np.arange(n_spec)

    # Polynomial fit to the spectral shift between spectrum
    x_mean = np.arange(len(l_shift_corr))
    fit_mean_poly = np.polyfit(x_mean, l_shift_corr, deg=2)
    gpol = np.poly1d(fit_mean_poly)
    y_model_pol = np.linspace(-5, 25, 100)
    x_model_pol = pos_wl0 + gpol(y_model_pol)

    fit = [x_fitted, y_spectrum, label_fitted]
    polyn_model = [x_model_pol, y_model_pol, fit_mean_poly]
    return l_shift_corr, fit, polyn_model, gpol(y_spectrum)


def compute_master_ref(spectra_align, wl_align, l_shift_corr, n_box=50):
    spectra_align_interp = apply_shift_fourier(spectra_align, l_shift_corr)

    ref_spectra_shifted = spectra_align_interp.mean(axis=0)

    master_ref = _substract_run_med(ref_spectra_shifted, wave=wl_align, n_box=n_box)[0]
    return master_ref


def apply_shift_fourier(l_spec, l_shift):
    """Apply the shift (from correlation map fit) using
    spline interpolation (3rd order). """
    n_spec = l_spec.shape[0]
    l_spec_align = []
    for i in range(n_spec):
        spec = l_spec[i]
        spec_shifted = shift(spec, l_shift[i])
        l_spec_align.append(spec_shifted)
    l_spec_align = np.array(l_spec_align)
    return l_spec_align


def fit_tellu_line(
    index, wave, spec, size=5, right=True, smart=True, abs_err=0.1, display=False
):
    tellu_wl0 = tellu[index]

    delta_wl = size * np.diff(wave).mean()
    cond_tellu = (tellu_wl0 - delta_wl <= wave) & (wave <= tellu_wl0 + delta_wl)

    spectr_around_tellu = -spec[cond_tellu]
    spectr_around_tellu -= spectr_around_tellu.min()
    wl_around_tellu = wave[cond_tellu]

    if smart:
        if right:
            spectr_around_tellu[
                wl_around_tellu <= tellu_wl0 - 0.0003
            ] = spectr_around_tellu.min()
        else:
            spectr_around_tellu[
                wl_around_tellu >= tellu_wl0 + 0.0003
            ] = spectr_around_tellu.min()

    spectr_around_tellu /= spectr_around_tellu.max()

    g_init = models.Gaussian1D(
        amplitude=spectr_around_tellu.min(), mean=tellu_wl0, stddev=0.002
    )
    yerr = np.ones(len(wl_around_tellu)) * abs_err
    fit_g = fitting.LevMarLSQFitter()
    g = fit_g(g_init, wl_around_tellu, spectr_around_tellu, weights=1.0 / yerr ** 2)

    chi2 = np.sum(
        (abs(spectr_around_tellu ** 2 - g(wl_around_tellu) ** 2) / yerr ** 2)
    ) / (len(wl_around_tellu) - 2)

    x_model = np.linspace(wl_around_tellu[0], wl_around_tellu[-1], 100)
    wl_find_gauss = g.mean.value
    offset_find = (wl_find_gauss - tellu_wl0) * 1000.0

    data = [wl_around_tellu, spectr_around_tellu, yerr]
    model = [x_model, g(x_model)]

    if display:
        plt.figure()
        plt.errorbar(
            data[0],
            data[1],
            yerr=yerr,
            marker="o",
            ls="None",
            color="#1560bd",
            label="data",
        )
        plt.plot(
            model[0],
            model[1],
            color="orange",
            lw=1,
            label=r"$\lambda_{off}$ = %2.3f nm ($\chi2$ = %2.1f)" % (offset_find, chi2),
        )
        plt.axvline(
            tellu_wl0,
            color="crimson",
            lw=1,
            label=r"Telluric line (%2.4f $\mu$m/%i)" % (tellu_wl0, index),
        )
        plt.legend()
        plt.xlabel(r"Wavelength [$\mu$m]")
        plt.ylabel("Relative flux [count]")
        plt.grid(alpha=0.2)

    return offset_find, data, model, chi2


def fit_tellu_offset(
    spec, wave, size=5, right=True, smart=True, abs_err=0.1, lim_chi2=10, verbose=False
):
    l_offset, l_chi2, l_res = [], [], []
    which_line = [0, 4, 8, 16, 23, 25]  # range(len(tellu) - 1)
    for i in which_line:
        result_fit = fit_tellu_line(
            i, wave, spec, size=size, abs_err=abs_err, right=right, smart=smart
        )
        l_res.append(result_fit)
        l_offset.append(result_fit[0])
        l_chi2.append(result_fit[3])
    l_offset = np.array(l_offset)
    l_chi2 = np.array(l_chi2)
    l_res = np.array(l_res, dtype=object)

    good_pts = l_chi2 >= lim_chi2
    mean_offset = l_offset[~good_pts].mean()
    if verbose:
        print("Found offset using tellurics = %2.3f nm" % mean_offset)
    res = {
        "l_offset": l_offset,
        "l_chi2": l_chi2,
        "l_res": l_res,
        "offset": mean_offset / 1000.0,
        "lines": which_line,
    }
    return res


def write_wave(calib_wave_file, shift, tellu_offset=0):
    """Rewrite the _wave file with applied correction
    and wavelength calibration. `shift` is computed 
    using correlation map, `tellu_offset` is computed using 
    gaussian fitting on tellurics.
    """
    fitsHandler = fits.open(calib_wave_file)
    if "_backup" not in calib_wave_file:
        file_backup = calib_wave_file.split(".fits")[0] + "_backup.fits"
    else:
        file_backup = calib_wave_file
        calib_wave_file = calib_wave_file.split("_backup")[0] + ".fits"
        cprint(
            "Warning: _wave_backup (not modified) is used to rewrite the _wave.fits.",
            "green",
        )

    fitsHandler[0].header["HIERARCH ESO PRO CATG"] = "WAVE_BACKUP"
    fitsHandler.writeto(file_backup, overwrite=True)

    for hdu in fitsHandler[1:]:
        if hdu.header["EXTNAME"] == "WAVE_DATA_SC":
            for i in np.arange(1, 25):
                datai = np.squeeze(hdu.data["DATA%i" % i])
                datai += shift[i - 1] / 1e6
                datai -= tellu_offset / 1e6
                hdu.data["DATA%i" % i] = datai
    fitsHandler.writeto(calib_wave_file, overwrite=True)
    return None
