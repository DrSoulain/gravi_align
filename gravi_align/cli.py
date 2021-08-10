from argparse import ArgumentParser
from typing import List, Optional

from gravi_align.api import (
    check_align_gravity,
    perform_align_gravity,
    compute_p2vm,
    clean_reduced_dir,
)


def main(argv: Optional[List[str]] = None) -> int:
    parser = ArgumentParser()

    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    run_parser = subparsers.add_parser("run", help="Run gravity alignment calibration")
    run_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Use the backup if any and recompute the shift (should not be used first).",
    )
    run_parser.add_argument(
        "-s", "--save", action="store_true", help="Save figures as pdf."
    )
    run_parser.add_argument(
        "--poly", action="store_true", help="Use the polynomial fit as shifts."
    )
    run_parser.add_argument(
        "--full",
        action="store_true",
        help="Compute the correlation map using large wavelength range (2.1-2.2). "
        + "Otherwise, correlation is performed on the telluric doublet around 2.18 microns.",
    )
    run_parser.add_argument(
        "--smooth", default=1, type=int, help="Kernel size to smooth the signal."
    )

    run_parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Show the figures (if any display available).",
    )
    run_parser.add_argument(
        "-d",
        "--default",
        action="store_true",
        help="Use the first calibrator as reference",
    )
    run_parser.add_argument(
        "-w",
        "--write",
        action="store_true",
        help="Overwrite the _wave file (in reduced/) and apply shift",
    )
    run_parser.add_argument(
        "--datadir",
        default="reduced/",
        help="Data directory to find _wave and _spectrumaligned fits (default: %(default)s)",
    )

    check_parser = subparsers.add_parser(
        "check",
        help="Apply the save shift/offset (save_shift.fits) to check the alignment",
    )
    check_parser.add_argument(
        "--wl",
        nargs="+",
        default=[2.184, 0.003],
        type=float,
        help="Range of wavelengths to plot the normalized spectra.",
    )

    check_parser.add_argument(
        "--datadir",
        default="reduced/",
        help="Data directory to find _wave and _spectrumaligned fits (default: %(default)s)",
    )
    check_parser.add_argument(
        "-d",
        "--default",
        action="store_true",
        help="Check the alignment on the first calibrator",
    )
    check_parser.add_argument(
        "-s", "--save", action="store_true", help="Save the figures in fig/"
    )

    p2vm_parser = subparsers.add_parser(
        "p2vm", help="Modify the sof and compute the p2vm with the new _wave",
    )
    p2vm_parser.add_argument(
        "--datadir",
        default="reduced/",
        help="Data directory to find .sof files (default: %(default)s)",
    )

    clean_parser = subparsers.add_parser(
        "clean", help="Clean the reduced/ dir and keep only calibration files",
    )
    clean_parser.add_argument(
        "--datadir",
        default="reduced/",
        help="Data directory to clean default: %(default)s)",
    )

    args = parser.parse_args(argv)

    # value = check_parser.parse_args()
    # print(value)

    if args.command == "run":
        perform_align_gravity(args)
    elif args.command == "check":
        check_align_gravity(args)
    elif args.command == "p2vm":
        compute_p2vm(args)
    elif args.command == "clean":
        clean_reduced_dir(args)

    return 0


if __name__ == "__main__":
    exit(main())
