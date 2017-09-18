#!/usr/bin/env python3
"""
This command line tool is used to start the python code analysis. For this purpose it uses pep8 for coding conventions
and pylint for extended issues.
"""
from sys import argv, executable
from argparse import ArgumentParser
from subprocess import run
from fnmatch import filter as fn_filter
from os import walk, path


def main(arguments):
    """
    This method starts the execution of the code analysis on every python file of a given path.
    :param arguments: [String, int]
                 This may include the path to the current path to analyse and the maximal number of characters
                 for each line of code.
    """
    parser = ArgumentParser(
        usage='This tool can be used to check the python code in a directory on style\nviolations.'
              'The scripts default path is the directory in which this script is\nlocated in.'
    )
    parser.add_argument(
        '--path', help='a path to the directory where the python code should be analyzed '
        '(default path os ".")', default='.', type=str, dest='path',
    )
    parser.add_argument(
        '--max-line-length', help='maximal number of characters for each line (default is 120)',
        default='120', type=int, dest='max_line_length',
    )
    args = parser.parse_args(arguments)

    # set maximal line length
    max_line_length = '--max-line-length={0}'.format(args.max_line_length)
    # run pep8 check
    run([executable, '-m', 'pep8', max_line_length, args.path])

    # pylint does not support to check subdirectories by default
    files_to_check = []
    for root, _, file_names in walk(args.path):
        for filename in fn_filter(file_names, '*.py'):
            files_to_check.append(path.join(root, filename))

    # if the path is a single file
    if not files_to_check:
        files_to_check = [args.path]

    pylint_cmd = [executable, '-m', 'pylint', max_line_length, '--disable=R']
    pylint_cmd.extend(files_to_check)
    # run pylint check
    run(pylint_cmd)


if __name__ == '__main__':
    main(argv[1:])
