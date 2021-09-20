# Spectral calibration/alignment of GRAVITY

[![Python 3.9](https://img.shields.io/pypi/pyversions/amical)](https://www.python.org/downloads/release/python-370/)

Command-line tool to perform a spectral calibration on high-resolution GRAVITY
data.

## Installation

It is recommended to create a separate environment with `conda create -n
<env_name> python=3.7`.
Then, within your Conda env (`conda activate <env_name>`):

```bash
# Install package from github
python -m pip install git+https://github.com/DrSoulain/gravi_align
```

## How to use it?

You are supposed to be on top repo where --datadir (default: reduced/)
repository is located. This calibration step supposed that you used
`run_gravi_reduced.py` first to save the spectrum files
(`*spectrumaligned.fits`), and created all `*.sof`, `*.sh` and `*_wave.fits`.

The first step uses the correlation map between spectrum to compute
inter-spectrum shifts. Spectral calibration is also performed to offset the entire
wavelengths table fitted with telluric lines around Brγ. You can add `-p` to plot diagnostic
figures and `-s` to save those figures (`fig_gravi_align/`). The diagnostic
figures include the computation of the applied shifts (in nm, 'compute_shift_\*.png'), the correlation
map ('corr_map_\*.png'), the raw ('raw_spectra_\*.png') and aligned spectrum ('aligned_spectra_\*.png'), the absolute
tellurics fit ('fit_tellu_\*.png') and the spectra plot
('Selected_spectrum_*.png'). 

> Note: If you used `--flag` argument,
'Selected_spectrum_*.png' show the rejected points for each spectrum.

You can add `-d`
to perform the calibration using the first calibrator file. See `gravi_align
run -h` for details.

```bash
gravi_align run
```

> Note: By default, the shifts are computed using the telluric doublet around 2.18 µm (2.182, 2.1865).

You can select an other region by adding `--corr` argument:

```bash
gravi_align run --corr 2.16 2.17
```

> Tips: It is usefull to check the region around Brγ (i.e.: --corr 2.146 2.186).

By default, the Brγ line is skipped by the correlation to focus on telluric lines. For noisy data, where tellurics are too weak, you can
force to use the Brγ region by adding `--brg`.

Once good and satisfied by the alignment, you can overwrite the `*_wave.fits`
with the same arguments (-corr, --flag, --brg, etc. if any):

```bash
gravi_align run -w
```

You can check the different spectrum (`*_spectrumaligned.fits`) alignment
present in --datadir (default: reduced/):

```bash
gravi_align check --wl 2.166 0.02
```


Then, you can modify the .sof corresponding to p2vm computation (add modified
`*_wave.fits` as `WAVE`) and run the p2vm recipe (esorex).

```bash
gravi_align p2vm
```

And finally, remove all the fits file in --datadir except the calibration files.
The diagnostic table is printed to check if all mandatory calibration files are present.

```bash
gravi_align clean
```

> Note: Be highly careful with this feature; the shifts computation and the
> associated `*_wave.fits` modification need to be executed before.

The calibrated `*_wave.fits` and the new associated p2vm calibration file
(`*_p2vm.fits`) are
now computed. You can now run `run_gravi_reduce.py` to finally extract your
data including the new spectral calibration.
