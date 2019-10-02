"""
Helper module to run studies as defined in pypuf.studies
"""
import argparse
import sys

from pypuf.experiments import Experimenter
from pypuf.tools import find_study_class


def main(args):
    """
    Tries to find the study specified and runs it.
    """
    parser = argparse.ArgumentParser(
        prog='study',
        description="Runs a pypuf study",
    )
    parser.add_argument("study", help="name of the study to be run", type=str)
    parser.add_argument(
        "--part",
        help="If specified, partitions the study into TOTAL parts and only runs PART (first part has index 0).",
        type=int,
        nargs=2,
        metavar=('PART', 'TOTAL'),
        default=[0, 1],
    )

    args = parser.parse_args(args)

    study_class = find_study_class(args.study)
    print(f'Running part {args.part[0]} of a total of {args.part[1]} parts of\n'
          f'{study_class.__module__}.{study_class.__name__}')
    study = study_class()
    study.run(part=args.part[0], total=args.part[1])


if __name__ == '__main__':
    Experimenter.disable_auto_multiprocessing()
    main(sys.argv[1:])
