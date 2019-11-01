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
        prog='info',
        description="Displays information about a pypuf study.",
    )
    parser.add_argument("study", help="name of the study to be run", type=str)

    args = parser.parse_args(args)

    study_class = find_study_class(args.study)
    print(f'{study_class.__module__}.{study_class.__name__}:')
    study = study_class()
    experiments = study.experiments()
    known_hashes = [ex.hash for ex in experiments]
    results = study.experimenter.results
    if not results.empty:
        unknown_experiments = results.loc[~results['experiment_hash'].isin(known_hashes)]
    else:
        unknown_experiments = []
    finished = 1 if not experiments else (len(results) - len(unknown_experiments)) / len(experiments)
    print(f'- total: {len(experiments)} experiments')
    print(f'- unknown: {len(unknown_experiments)} experiments')
    print(f'- finished: {len(results)} experiments ({finished * 100:.2f}%)')
    print(f'- pending: {len(experiments) - len(results) + len(unknown_experiments)} experiments')


if __name__ == '__main__':
    Experimenter.disable_auto_multiprocessing()
    main(sys.argv[1:])
