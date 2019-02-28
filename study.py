"""
Helper module to run studies as defined in pypuf.studies
"""
import importlib
import inspect
import sys
import argparse


def main(args):
    """
    Tries to find the study specified and runs it.
    """
    parser = argparse.ArgumentParser(
        prog='study',
        description="Runs a pypuf study",
    )
    parser.add_argument("study", help="name of the study to be run", type=str)

    args = parser.parse_args(args)

    if not args.study.startswith('pypuf.studies'):
        args.study = 'pypuf.studies.' + args.study

    try:
        study_module = importlib.import_module(args.study)
    except ModuleNotFoundError:
        print('Module {} cannot be found.'.format(args.study))
        exit(1)

    studies = [
        c[1] for c in inspect.getmembers(study_module, inspect.isclass)
        if isinstance(c, tuple) and len(c) > 1 and str(c[1].__module__).startswith(args.study)
    ]

    if not studies:
        print('Module {} does not contain any study.'.format(args.study))
        exit(1)

    if len(studies) > 1:
        print('Module {} contains more than one study:'.format(args.study))
        for s in studies:
            print(' - {}'.format(s))
        exit(1)

    study_class = studies[0]
    print('Running {}.{}'.format(study_class.__module__, study_class.__name__))
    study = study_class()
    study.run()


if __name__ == '__main__':
    main(sys.argv[1:])
