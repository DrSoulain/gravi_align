import numpy as np
from matplotlib import pyplot as plt
import pkg_resources

from gravi_align.core import _substract_run_med

# Plotting function
# ==============================================================================


def plot_tellu(label=None, plot_ind=False, val=5000):
    file_tellu = pkg_resources.resource_stream(
        "gravi_align", "internal_data/Telluric_lines.txt"
    )
    tellu = np.loadtxt(file_tellu, skiprows=1)
    plt.axvline(np.nan, lw=0.5, c="crimson", alpha=0.5, label=label)
    for i in range(len(tellu)):
        plt.axvline(tellu[i], lw=0.5, c="crimson", alpha=0.5)
        if plot_ind:
            plt.text(tellu[i], val, i, fontsize=7, c="crimson")


def plot_corr_map(
    corr_map, fit=None, polyn_model=None, window=25, save=False, figname=None
):
    """ Plot the correlation map with the detected offset and
    the polynomial fit. You can limit the region of interest
    using `window` (default = 25). """
    n_spec = corr_map.shape[0]
    n_channel = corr_map.shape[1]
    pos_wl0 = n_channel // 2

    fig = plt.figure(figsize=(14, 9))
    ax = plt.gca()
    ax.set_title("-- Correlation map --", fontsize=16)
    # plt.title('%s - %s - with p2vm %i' % (target, date_obs, p2vm))
    c = plt.imshow(corr_map, cmap="gist_earth", aspect="auto")
    if fit is not None:
        plt.plot(fit[0], fit[1], "x", color="#fff600", label="Gaussian fit positions")
        for i in range(n_spec):
            plt.text(fit[0][i], fit[1][i], s=fit[2][i], color="#fff600", fontsize=9)
    if polyn_model is not None:
        plt.plot(
            polyn_model[0],
            polyn_model[1],
            "w",
            ls="-",
            lw=1,
            label="Polynomial fit (%2.1e, %2.1e, %2.1e)"
            % (polyn_model[2][0], polyn_model[2][1], polyn_model[2][2],),
        )
    plt.legend(fontsize=12, facecolor="None", labelcolor="w")
    plt.xlim(pos_wl0 - window // 2, pos_wl0 + window // 2)
    plt.ylim(23.5, -0.5)
    plt.xlabel("Wavelength [pix]", fontsize=12)
    plt.ylabel("Spectrum numbers (on detector)", fontsize=12)
    cbar = plt.colorbar(c, ax=ax)
    cbar.set_label("Correlation", fontsize=12)
    plt.tight_layout()

    if save:
        if figname is None:
            figname = "correlation_map.pdf"
            plt.savefig(figname)
    return fig


def plot_spectra(
    l_spec,
    wave,
    l_offset=None,
    n_box=50,
    wl_lim=None,
    title="",
    f_range=None,
    div=False,
    save=False,
    display=False,
):
    n_spec = len(l_spec)
    if wl_lim is None:
        wl_lim = [2.16, 0.03]

    offset = 0
    fig = plt.figure(figsize=[6, 4])
    plt.title(title)
    norm_spec = []
    norm_wl = []
    for i in range(n_spec):
        spec, wave_b = _substract_run_med(
            l_spec[i], wave=wave, n_box=n_box, div=div, display=display
        )
        if l_offset is not None:
            try:
                offset = l_offset[i]
            except IndexError:
                offset = l_offset
        plt.plot(wave_b + offset, spec, ",-")
        norm_spec.append(spec)
        norm_wl.append(wave_b + offset)

    spec_mean = np.median(norm_spec, axis=0)
    plt.plot(wave_b, spec_mean, "k--", lw=1)
    plot_tellu()
    plt.xlim(wl_lim[0] - wl_lim[1], wl_lim[0] + wl_lim[1])
    if f_range is not None:
        plt.ylim(f_range)
    plt.xlabel(r"Wavelength [$\mu$m]")
    plt.ylabel("Normalized flux [counts]")
    plt.tight_layout()
    if save:
        plt.savefig("figures/" + title.replace(" ", "_") + ".png")
    return fig


def plot_tellu_fit(res, lim_chi2=10):
    file_tellu = pkg_resources.resource_stream(
        "gravi_align", "internal_data/Telluric_lines.txt"
    )
    tellu = np.loadtxt(file_tellu, skiprows=1)

    l_offset = res["l_offset"]
    l_res = res["l_res"]
    l_chi2 = res["l_chi2"]
    lines = res["lines"]

    fig = plt.figure(figsize=[12, 6])
    fig.suptitle(
        r"Fit tellurics around Br$\gamma$ - offset=%2.3fnm" % (res["offset"] * 1000.0),
        fontsize=16,
    )
    for i in range(len(l_offset)):
        res = l_res[i]
        chi2 = l_chi2[i]
        x_data, y_data, err_data = res[1]
        x_mod, y_mod = res[2]
        plt.subplot(3, 2, i + 1)
        plt.errorbar(
            x_data,
            y_data,
            yerr=err_data,
            marker=".",
            ls="None",
            color="#1560bd",
            label="data",
        )

        plt.plot(
            x_mod,
            y_mod,
            color="orange",
            lw=1,
            label=r"$\lambda_{off}$=%2.3fnm ($\chi2$=%2.1f)" % (l_offset[i], l_chi2[i]),
        )
        tellu_wl0 = tellu[lines[i]]
        plt.axvline(
            tellu_wl0,
            color="g",
            lw=1,
            label=r"Telluric line (%2.4f $\mu$m)" % (tellu_wl0),
        )
        if chi2 > lim_chi2:
            plt.plot(tellu_wl0, 0.5, "rx", ms=20)
        plt.legend(fontsize=5)
    plt.tight_layout()
    return fig


def plot_raw_spectra(
    spectra,
    wave_cal,
    shift=None,
    tellu_offset=0,
    wl_lim=[2.185, 0.003],
    title=None,
    args=None,
):
    """ """
    n_spec = spectra.shape[0]

    fig = plt.figure(figsize=[9, 6])
    plt.title(title, fontsize=16)
    for i in range(n_spec):
        spec, wave = _substract_run_med(spectra[i], wave_cal[i])
        if shift is None:
            tmp_shift = 0
        else:
            tmp_shift = shift[i]

        wl_fixed = (wave * 1e6) + tmp_shift - tellu_offset

        if args is not None:
            cond_wl = (wl_fixed >= args.corr[0]) & (wl_fixed <= args.corr[1])
        else:
            cond_wl = (wl_fixed >= wl_lim[0] - wl_lim[1]) & (
                wl_fixed <= wl_lim[0] + wl_lim[1]
            )
        plt.plot(wl_fixed[cond_wl], spec[cond_wl])
    if args is not None:
        plt.xlim(args.corr[0], args.corr[1])
    else:
        plt.xlim(wl_lim[0] - wl_lim[1], wl_lim[0] + wl_lim[1])
    plot_tellu()
    plt.xlabel(r"Wavelength [$\mu$m]")
    plt.ylabel("Normalized flux [counts]")
    plt.tight_layout()
    return fig
