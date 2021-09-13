from glob import glob
import time
import os

from matplotlib import pyplot as plt
from gravi_align.plotting import plot_corr_map, plot_raw_spectra, plot_tellu_fit
from gravi_align.core import (
    compute_corr_map,
    compute_master_ref,
    compute_shift,
    fit_tellu_offset,
    open_spectrum_file,
    write_wave,
)
from astropy.io import fits
import numpy as np
from termcolor import cprint
from scipy.constants import c as c_light


def find_wave(args):
    if not os.path.exists(args.datadir):
        raise IOError("Datadir %s/ not found, check --datadir argument." % args.datadir)

    try:
        calib_wave_file = glob("%s/*_wave.fits" % (args.datadir))[0]
    except IndexError:
        raise IndexError("*_wave.fits not found in %s/." % args.datadir)

    l_backup = glob("%s/*_wave_backup.fits" % (args.datadir))
    if len(l_backup) == 1:
        cprint(
            "Warning: _wave_backup found, shift are already applied (you can add -f to recompute the shifts).",
            color="green",
        )
        calib_wave_file = l_backup[0]
    data_wave = fits.getdata(calib_wave_file, "WAVE_DATA_SC")
    wave_cal = np.array([data_wave["DATA%i" % i].sum(axis=0) for i in np.arange(1, 25)])
    return wave_cal, calib_wave_file


def load_data(args):
    l_file = glob("%s/*aligned.fits" % args.datadir)
    if len(l_file) == 0:
        raise IOError(
            "No spectra files detected in %s, check --datadir. Did you use run_gravi_reduce.py first?"
            % args.datadir
        )

    if not args.default:
        print("\n  SOURCE  |  DATE-OBS  |  index  ")
        print("-----------------------------------")
    else:
        print("First calibrator found used by default:")
    l_cal = []
    for i, f in enumerate(l_file):
        hdu = fits.open(f)
        target = hdu[0].header["OBJECT"]
        date = hdu[0].header["DATE-OBS"]
        sci = hdu[0].header["HIERARCH ESO PRO SCIENCE"]
        if not sci:
            l_cal.append(i)
        if not args.default:
            print("  %s     %s     %s     %i" % (target, date, sci, i))

    if args.default:
        choosen_index = l_cal[0]
    else:
        if args.command == "run":
            choosen_index = int(input("Which file is the reference?\n"))
        elif args.command == "check":
            choosen_index = int(input("Which file you want to check?\n"))

    if args.command == "run":
        s = "Reference"
    elif args.command == "check":
        s = "Checked"

    try:
        sel_ref = fits.open(l_file[choosen_index])[0].header["OBJECT"]
        obs_ref = fits.open(l_file[choosen_index])[0].header["DATE-OBS"]
        cprint("-> %s file %s (%s)" % (s, sel_ref, obs_ref), "cyan")

        filename = l_file[choosen_index]
        spectra, wl_align, spectra_align = open_spectrum_file(filename)
    except IndexError:
        raise IndexError(
            "Selected index (%i) not valid (only %i files found)."
            % (choosen_index, len(l_file))
        )

    return spectra, wl_align, spectra_align, sel_ref, obs_ref


def perform_align_gravity(args):
    print("\n -------- Start GRAVITY spectral alignment --------")
    start_time = time.time()

    wave, wavefile = find_wave(args)

    if ("_backup" in wavefile) & (not args.force):
        print(
            " -------- Spectral alignment done (%2.2f s) -------- \n"
            % (time.time() - start_time)
        )
        return 0

    spectra, wl_align, spectra_align, sel_ref, obs_ref = load_data(args)
    t1 = time.time()
    print("[1] Load files (%2.2f s)" % (t1 - start_time))
    # Size of the box to compute the running median and normalize the spectra
    n_spec = spectra_align.shape[0]

    corr_lim = args.corr
    if args.full:
        corr_lim = [2.1, 2.2]

    # Compute the cross-correlation function as 2D array
    corr_map = compute_corr_map(
        spectra_align, wl_align, corr_lim=corr_lim, smooth=args.smooth
    )
    t2 = time.time()
    print("[2] Compute correlation map (%2.2f s)" % (t2 - t1))

    # Compute the shift using gaussian model and fit 2nd order polynom
    # to the results
    shift = compute_shift(corr_map)

    i_fit = 0
    if args.poly:
        i_fit = 3

    pixel_lambda = np.diff(wl_align).mean()
    master_ref = compute_master_ref(spectra_align, wl_align, shift[i_fit])
    # Compute the cross-correlation function as 2D array
    corr_map = compute_corr_map(
        spectra_align,
        wl_align,
        master_ref=master_ref,
        corr_lim=corr_lim,
        smooth=args.smooth,
    )
    new_shift = compute_shift(corr_map)

    if args.save:
        if not os.path.exists("fig_gravi_align/"):
            os.mkdir("fig_gravi_align")

    plot_corr_map(corr_map, new_shift[1], new_shift[2])
    if args.save:
        plt.savefig("fig_gravi_align/corr_map_%s_%s.png" % (sel_ref, obs_ref))

    computed_shift = pixel_lambda * new_shift[i_fit]
    std_shift = pixel_lambda * new_shift[4]

    std_shift_vel = (std_shift.mean() / args.restframe) * c_light / 1e3

    plt.figure(figsize=[9, 6])
    plt.title(
        r"Spectral shift applied - $\Delta\lambda_{m}$ = %2.2f km/s @ %2.3f µm"
        % (std_shift_vel, args.restframe),
        fontsize=16,
    )
    plt.errorbar(
        np.arange(len(computed_shift)),
        computed_shift * 1e3,
        yerr=std_shift * 1e3,
        label="Fit uncertainty = %2.3f nm" % (1e3 * std_shift.mean()),
    )
    plt.legend()
    plt.xlabel("# Spectrum")
    plt.ylabel("Spectral shift [nm]")
    plt.grid(alpha=0.2)
    plt.tight_layout()

    t3 = time.time()
    print("[3] Compute shifts between spectrum (%2.2f s)" % (t3 - t2))

    res = fit_tellu_offset(
        spectra_align.mean(axis=0), wl_align, lim_chi2=4, verbose=True
    )
    computed_offset = res["offset"]
    t4 = time.time()
    print("[4] Compute offset from tellurics (%2.2f s)" % (t4 - t3))

    plot_tellu_fit(res, lim_chi2=4)
    if args.save:
        plt.savefig("fig_gravi_align/fit_tellu_%s_%s.png" % (sel_ref, obs_ref))

    if args.write:
        write_wave(
            wavefile, shift=computed_shift, tellu_offset=computed_offset,
        )
        t5 = time.time()
        print("[5] Overwrite _wave.fits (%2.2f s)" % (t5 - t4))

    fits.writeto(
        "save_shift.fits",
        np.array([computed_shift, n_spec * [computed_offset]]),
        overwrite=True,
    )

    wl_lim = [2.184, 0.003]
    plot_raw_spectra(
        spectra,
        wave,
        shift=None,
        tellu_offset=0,
        wl_lim=wl_lim,
        title="Raw spectra - %s %s" % (sel_ref, obs_ref),
        args=args,
    )
    if args.save:
        plt.savefig(
            "fig_gravi_align/raw_spectra_%s_%s.png" % (sel_ref, obs_ref), dpi=300
        )

    plot_raw_spectra(
        spectra,
        wave,
        shift=computed_shift,
        tellu_offset=computed_offset,
        wl_lim=wl_lim,
        title="Aligned spectra - %s %s" % (sel_ref, obs_ref),
        args=args,
    )
    if args.save:
        plt.savefig(
            "fig_gravi_align/aligned_spectra_%s_%s.png" % (sel_ref, obs_ref), dpi=300
        )

    print(
        " -------- Spectral alignment done (%2.2f s) -------- \n"
        % (time.time() - start_time)
    )
    if args.plot:
        plt.show()
    return 0


def check_align_gravity(args):
    print(" -------- Check GRAVITY spectral alignment --------")
    wave, wavefile = find_wave(args)
    spectra, wl_align, spectra_align, sel_ref, obs_ref = load_data(args)

    if args.save:
        if not os.path.exists("fig_gravi_align/"):
            os.mkdir("fig_gravi_align")

    try:
        computed_shift, computed_offset = fits.open("save_shift.fits")[0].data
        computed_offset = computed_offset[0]
    except FileNotFoundError:
        cprint("save_shift.fits not found: do 'gravi_align run' first!", "red")
        computed_shift = None
        computed_offset = 0

    s1, s2 = "not aligned", "not shifted"
    if computed_shift is not None:
        s1 = "aligned"
    if computed_offset != 0:
        s2 = "shifted"
    s = "%s/%s" % (s1, s2)

    title = "Data file %s (%s): %s" % (sel_ref, obs_ref, s)
    plot_raw_spectra(
        spectra,
        wave,
        shift=computed_shift,
        tellu_offset=computed_offset,
        wl_lim=args.wl,
        title=title,
    )
    if args.lim is not None:
        plt.ylim(args.lim[0][0], args.lim[0][1])

    if args.save:
        plt.savefig("fig_gravi_align/check_spectra_%s_%s.pdf" % (sel_ref, obs_ref))
    plt.show()

    return 0


def _find_p2vm_sof(list_sof):
    for f in list_sof:
        d = np.loadtxt(f, dtype=str)
        if ("P2VM_RAW" in d) & ("WAVE_RAW" in d) & ("WAVESC_RAW" in d):
            p2vm_sof = f
    return p2vm_sof


def _check_p2vm_modified(p2vm_sof):
    for ss in p2vm_sof:
        if "_wave" in ss[0]:
            cprint(
                "Warning: p2vm .sof seems to be already modified (%s found)" % ss[0],
                "green",
            )
            return True


def compute_p2vm(args):
    calib_wave_file = glob("%s/*_wave.fits" % (args.datadir))[0]
    datadir = args.datadir
    list_sof = glob(datadir + "*.sof")

    file_p2vm_sof = _find_p2vm_sof(list_sof)
    p2vm_sof = np.loadtxt(file_p2vm_sof, dtype=str)

    log_file_name = file_p2vm_sof.split(".sof")[0].split("/")[1]
    sof_file_name = "new_p2vm_%s.sof" % log_file_name
    if _check_p2vm_modified(p2vm_sof):
        return 0
    else:
        new_line = np.array([calib_wave_file, "WAVE"], dtype=str)
        new_p2vm = np.array(np.vstack([p2vm_sof, new_line]), dtype=str)
        np.savetxt(sof_file_name, new_p2vm, fmt="%s")
        os.system(
            "esorex --log-file=new_p2vm_%s.log --output-dir=%s gravity_p2vm %s"
            % (log_file_name, args.datadir, sof_file_name)
        )

    return 0


def clean_reduced_dir(args):
    list_file = glob(args.datadir + "*")
    if len(list_file) == 0:
        cprint(
            "Warning: %s seems to be empty, check --datadir (nothing removed)."
            % args.datadir,
            "green",
        )
        return 0

    list_bool = np.array([False, False, False, False, False])
    list_calib = np.array(["_wave.", "_dark.", "_p2vm.", "_flat.", "_bad."])
    list_found_cal = []
    for f in list_file:
        is_calib = np.array([x in f for x in list_calib])
        list_bool[is_calib] = True
        if True in is_calib:
            list_found_cal.append(f)
        else:
            os.remove(f)
    print("------ Calibration found -------")
    print(" wave | dark | p2vm | flat | bad")
    dic_sign = {"True": "o", "False": "x"}
    list_sign = [dic_sign[str(x)] for x in list_bool]
    print(
        "  %s   |   %s  |   %s  |   %s  |  %s   "
        % (list_sign[0], list_sign[1], list_sign[2], list_sign[3], list_sign[4])
    )
    print("--------------------------------")
    return 0
