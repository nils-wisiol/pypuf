#!/usr/bin/env python3
"""
This command line tool is used to start the python code analysis. For this purpose it uses pep8 for coding conventions
and pylint for extended issues.
"""
from sys import argv, executable
from argparse import ArgumentParser
from subprocess import call
from fnmatch import filter as fn_filter
from os import walk, path
from functools import reduce


def main(arguments):
    """
    This method starts the execution of the code analysis on every python file of a given path.
    :param arguments: [String, int]
                 This may include the path to the current path to analyse, the maximal number of characters
                 for each line of code or some patterns to skip certain paths.
    """
    parser = ArgumentParser(
        usage='This tool can be used to check the python code in a directory on style\nviolations.'
              'The scripts default path is the directory in which this script is\nlocated in.'
    )
    parser.add_argument(
        '-p', '--path', help='a path to the directory where the python code should be analyzed '
                             '(default path os ".")', default='.', type=str, dest='path',
    )
    parser.add_argument(
        '-l', '--max-line-length', help='maximal number of characters for each line (default is 120)',
        default='120', type=int, dest='max_line_length',
    )
    parser.add_argument(
        '-e', '--exclude_patterns', help='patterns to identify directories or files which should not be checked '
                                         '(default are env .env)',
        default=['.env', 'env', 'venv', '.venv'], type=str, dest='exclude_patterns', nargs='+',
    )
    args = parser.parse_args(arguments)

    # set maximal line length
    max_line_length = '--max-line-length={0}'.format(args.max_line_length)

    # set paths which should not be checked
    excludes = '--exclude={0}'.format(','.join(args.exclude_patterns))

    # run pycodestyle check
    pycodestyle_returncode = call([executable, '-m', 'pycodestyle', max_line_length, excludes, args.path])

    # exclude specific directories support to check subdirectories by default
    files_to_check = []
    for root, _, file_names in walk(args.path):
        # skip certain path which includes some certain patterns
        if reduce((lambda x, y: x or y), [pattern in root for pattern in args.exclude_patterns]):
            continue

        for filename in fn_filter(file_names, '*.py'):
            if not reduce((lambda x, y: x or y), [pattern in filename for pattern in args.exclude_patterns]):
                files_to_check.append(path.join(root, filename))

    # if the path is a single file
    if not files_to_check:
        files_to_check = [args.path]

    pylint_cmd = [executable, '-m', 'pylint', max_line_length, '--disable=R']
    pylint_cmd.extend(files_to_check)
    # run pylint check
    pylint_returncode = call(pylint_cmd)

    returncode = 0
    if pycodestyle_returncode != 0 or pylint_returncode != 0:
        returncode = 1

    exit(returncode)


if __name__ == '__main__':
    main(argv[1:])
