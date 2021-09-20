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
from tsmoothie.smoother import ConvolutionSmoother

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
    e_flux_align = np.array(
        [data["DATAERR%i" % i].sum(axis=0) for i in np.arange(1, nint)]
    )

    try:
        wave_align = fits.open(file)["OI_WAVELENGTH", 10].data.field("EFF_WAVE") * 1e6
    except KeyError:
        wave_align = fits.open(file)["OI_WAVELENGTH", 11].data.field("EFF_WAVE") * 1e6

    return flux, wave_align, flux_align, e_flux_align


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


def _substract_run_med(spectrum, wave=None, err=None, n_box=50, shift_wl=0, div=False):
    """ Substract running median from a raw spectrum `f`. The median
    is computed at each points from n_box/2 to -n_box/2+1 in a
    'box' of size `n_box`. The Br gamma line in vaccum and telluric
    lines can be displayed if wavelengths table (`wave`) is specified.
    `shift_wl` can be used to shift wave table to estimate the
    spectral shift w.r.t. the telluric lines.
    """
    r_med = _running_median(spectrum, n_box)
    boxed_flux = spectrum[n_box // 2 : -n_box // 2 + 1]

    boxed_wave = np.arange(len(boxed_flux))
    boxed_err = np.zeros_like(boxed_flux)

    if wave is not None:
        boxed_wave = wave[n_box // 2 : -n_box // 2 + 1] - shift_wl

    if err is not None:
        boxed_err = err[n_box // 2 : -n_box // 2 + 1]

    boxed_err = np.array(boxed_err)

    res = boxed_flux - r_med
    if div:
        res = boxed_flux / r_med

    return res, boxed_wave, boxed_err


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
    l_shift_std = np.zeros(n_spec)
    for i_fit in range(n_spec):
        cond_sel = (x_corr < size) & (x_corr > -size)
        x_to_fit = x_corr[cond_sel]
        y_to_fit = corr_map[i_fit][cond_sel]
        g_init = models.Gaussian1D(amplitude=y_to_fit.max(), mean=1, stddev=1.0)
        g = fitter_gauss(g_init, x_to_fit, y_to_fit)
        l_shift_corr[i_fit] = g.mean.value
        cov_diag = np.diag(fitter_gauss.fit_info["param_cov"])
        l_shift_std[i_fit] = np.sqrt(cov_diag[1])

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
    return l_shift_corr, fit, polyn_model, gpol(y_spectrum), l_shift_std


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


def compute_sel_spectra(
    spectra_align, wl_align, e_spectra, corr, nbox=50, sigma=5, use_flag=True
):
    """ Normalize and select spectrum around a specified correlation region (by
    default around 2.185 µm telluric doublet [`corr`]). Compute flag in data point are too
    far from the 5-sigma [`sigma`] limits (averaged standard dev between spectra)."""
    # cc = plt.cm.turbo(np.linspace(0, 1, len(spectra_align)))

    sel_flux, sel_wl, sel_std, sel_err = [], [], [], []
    for i in range(len(spectra_align)):
        inp_spectre = spectra_align[i]
        tmp = _substract_run_med(
            inp_spectre, wave=wl_align, n_box=nbox, err=e_spectra[i]
        )
        cond_wl2 = (tmp[1] >= corr[0]) & (tmp[1] <= corr[1])
        wl = tmp[1][cond_wl2]
        flux = tmp[0][cond_wl2]
        err = tmp[2][cond_wl2]
        std = sigma * flux.std()
        sel_std.append(std)
        sel_err.append(err)
        sel_flux.append(flux)
        sel_wl.append(wl)
    master_std = np.mean(sel_std)
    master_wl = sel_wl[0]

    sel_flux = np.array(sel_flux)
    sel_std = np.array(sel_std)
    sel_err = np.array(sel_err)

    sel_flag = []
    for i in range(len(sel_flux)):
        flux = sel_flux[i] + i * 4 * master_std
        aver = np.mean(flux)
        cond_flag = (flux >= aver - 1.5 * master_std) & (flux <= aver + 1 * master_std)
        sel_flag.append(cond_flag)
    sel_flag = np.array(sel_flag)

    master_wl_backup = master_wl.copy()

    if use_flag:
        master_flag = np.mean(sel_flag, axis=0) == 1
        master_wl = master_wl[master_flag]

        sel_flux_flag = np.zeros([len(sel_flux), len(master_wl)])
        sel_err_flag = np.zeros_like(sel_flux_flag)
        for i in range(len(sel_flux)):
            sel_flux_flag[i] = sel_flux[i][master_flag]
            sel_err_flag[i] = sel_err[i][master_flag]
        master_spectrum = np.mean(sel_flux_flag, axis=0)
    else:
        master_flag = np.zeros(len(sel_flux)) == 0
        master_spectrum = np.mean(sel_flux, axis=0)
        sel_flux_flag = sel_flux
        sel_err_flag = sel_err

    ymin = np.mean(master_spectrum - 25 * 4 * master_std)
    ymax = np.mean(master_spectrum + 2 * 4 * master_std)

    plt.figure(figsize=(6, 8))
    plt.title("GRAVITY 24 spectra", fontsize=16)
    for i in range(len(sel_flux)):
        flux = sel_flux[i] - i * 4 * master_std
        aver = np.mean(flux)
        cond_flag = (flux >= aver - 1.5 * master_std) & (flux <= aver + 1 * master_std)
        plt.plot(master_wl_backup, flux, lw=1, color="gray")
        plt.scatter(
            master_wl_backup,
            flux,
            c=flux,
            zorder=10,
            s=25,
            marker=".",
            cmap="coolwarm_r",
        )
        plt.plot(master_wl_backup[~cond_flag], flux[~cond_flag], "rx", zorder=20, ms=10)
    plt.plot(
        master_wl,
        master_spectrum + 1 * 4 * master_std,
        "k+-",
        lw=2,
        label="Master spectra",
    )
    plt.plot(np.nan, np.nan, "rx", label="Flagged")
    plt.ylim(ymin, ymax)
    plt.legend(loc="best", fontsize=8)
    plt.xlabel("Wavelength [µm]")
    plt.ylabel("Flux [arbitrary unit]")
    plt.xlim(corr)
    plt.tight_layout()

    return master_spectrum, master_wl, sel_flux_flag, sel_err_flag, master_flag


def compute_corr_map(
    selected_spectra, master_ref=None, smooth=1, brg=[2.1623, 2.17], use_brg=True,
):
    """ Compute the 2D correlation map of several spectra.

    Parameters:
    -----------
    `selected_spectra` {array}:
        Spectrally selected and flagged array from `compute_sel_spectra()`,\n
    `master_ref` {array}:
        If any, master_ref is used as reference spectra,\n
    `smooth` {int}:
        Gaussian smoother size,\n
    `brg` {list}:
        Br$\gamma$ position [$\mu$m] to be excluded from the correlation,\n

    Outputs:
    --------
    `corr_map` {array}:
        Correlation matrix.
    """

    master_spectrum = selected_spectra[0]
    boxed_wave = selected_spectra[1]

    l_spec = selected_spectra[2]

    ref_spectrum = master_spectrum
    if master_ref is not None:
        ref_spectrum = master_ref

    smoother0 = ConvolutionSmoother(window_len=smooth, window_type="ones")
    smoother0.smooth(ref_spectrum)
    ref_spectrum = smoother0.smooth_data[0]

    n_spec = l_spec.shape[0]

    if not use_brg:
        cond_BrG = (boxed_wave >= brg[0]) & (boxed_wave <= brg[1])
    else:
        cond_BrG = boxed_wave > 0

    ref_spectrum_sel = ref_spectrum[~cond_BrG]
    size_spectr_norm = ref_spectrum_sel.shape[0]
    n_corr = (size_spectr_norm * 2) - 1

    l_spec_sub = []
    corr_map = np.zeros([n_spec, n_corr])

    for i in range(n_spec):
        spec_to_compare = l_spec[i]
        smoother1 = ConvolutionSmoother(window_len=smooth, window_type="ones")
        smoother1.smooth(spec_to_compare)
        spec_to_compare = smoother1.smooth_data[0]
        corr_tmp = correlate(ref_spectrum_sel, spec_to_compare[~cond_BrG])
        corr_map[i] = corr_tmp
        l_spec_sub.append(spec_to_compare[~cond_BrG])

    return corr_map
