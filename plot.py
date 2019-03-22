"""
Helper module to plot a study result as defined in pypuf.studies
"""
import sys
import argparse

from pypuf.experiments import Experimenter
from pypuf.tools import find_study_class


def main(args):
    """
    Tries to find the study specified and plots it.
    """
    parser = argparse.ArgumentParser(
        prog='study',
        description="Plot results from a pypuf study",
    )
    parser.add_argument("study", help="name of the study to be plotted", type=str)

    args = parser.parse_args(args)

    study_class = find_study_class(args.study)
    print('Plotting results for {}.{}'.format(study_class.__module__, study_class.__name__))
    study = study_class()
    study.plot()


if __name__ == '__main__':
    Experimenter.disable_auto_multiprocessing()
    main(sys.argv[1:])
