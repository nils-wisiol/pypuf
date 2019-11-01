"""
Helper module to run studies as defined in pypuf.studies
"""
import argparse
import logging
import os
import sys
from pprint import pprint
from queue import Queue

if 'PYPUF_CPU_LIMIT' in os.environ and os.environ['PYPUF_CPU_LIMIT'] != '1':
    raise ValueError('PYPUF_CPU_LIMIT must be unset or 1.')
else:
    os.environ['PYPUF_CPU_LIMIT'] = '1'

from pypuf.tools import find_study_class


def main(args):
    """
    Tries to find the study specified and runs it.
    """
    parser = argparse.ArgumentParser(
        prog='run',
        description="Runs an experiment of a pypuf study",
    )
    parser.add_argument("study", help="name of the study to be run", type=str)
    parser.add_argument("index", help="index of the experiment to be run", type=int)

    args = parser.parse_args(args)

    study_class = find_study_class(args.study)
    study = study_class()
    experiments = study.experiments()
    experiment = study.experiments()[args.index]
    print(f'Running experiment #{args.index}/{len(experiments)} of {study_class.__module__}.{study_class.__name__}')
    print('Parameters:')
    pprint(dict(experiment.parameters._asdict()))
    logging.basicConfig(level=logging.DEBUG)
    result = experiment.execute(Queue(), '')
    print('Result:')
    pprint(dict(result._asdict()))


if __name__ == '__main__':
    main(sys.argv[1:])
