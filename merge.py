"""
Helper module to run studies as defined in pypuf.studies
"""
import argparse
import sys

from pypuf.experiments import Experimenter


def main(args):
    """
    Tries to find the study specified and runs it.
    """
    parser = argparse.ArgumentParser(
        prog='merge',
        description="Merges several result files into a new one.",
    )
    parser.add_argument("source", help="files to be merged", nargs="+", type=str)
    parser.add_argument("dest", help="file to write", nargs=1, type=str)

    args = parser.parse_args(args)

    try:
        Experimenter.merge_result_files(args.source, args.dest[0])
    except FileNotFoundError as e:
        print(f'Could not find {e.filename.decode()}.')
        quit(1)


if __name__ == '__main__':
    main(sys.argv[1:])
