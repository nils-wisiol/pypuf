"""
This module provides some utility functions in oder to minimize duplicate code in tests.
"""
import os
import glob
import multiprocessing
from functools import wraps
from pypuf.experiments.experimenter import log_listener, setup_logger

LOG_PATH = 'test/'
LOG_NAME = 'test_log'


class TestLogger(object):
    """
    This class is used to provide a multiprocessing logger. Which is compatible with
    pypuf.experiments.experiment.base.Experiment.
    """
    def __init__(self, dir_name=LOG_PATH, log_name=LOG_NAME):
        assert dir_name[-1] == '/', 'dir_name must end with "/"'
        self.log_directory = dir_name
        self.logger_name = self.log_directory+log_name
        self.queue = None
        self.listener = None

    def start_multiprocessing_logger(self):
        """
        This method Setups multiprocessing logging.
        Only run one logger at a time.
        """
        assert self.listener is None, 'Logger is running.'
        assert self.queue is None, 'Queue should be None. You may use the class in a wrong way.'
        self.queue = multiprocessing.Queue(-1)
        self.listener = multiprocessing.Process(
            target=log_listener,
            args=(self.queue, setup_logger, self.logger_name,)
        )

        self.listener.start()

    def shutdown_multiprocessing_logger(self):
        """This method shutdown the multiprocessing logging"""
        self.queue.put_nowait(None)
        self.listener.terminate()
        self.listener = None
        self.queue = None


def logging(function):
    """
    This function provides a decorator which handles the logging for experiment functions.
    """
    logger = TestLogger(log_name=function.__name__)

    def _logging(*args, **kwargs):
        """
        This function starts and shutdowns the multiprocessing logger.
        :param args: extra positional generic arguments
        :param kwargs: extra keyword arguments generic argument
        :return: return value of `function`
        """
        logger.start_multiprocessing_logger()
        kwargs['logger'] = logger
        try:
            result = function(*args, **kwargs)
        except Exception as exception:
            raise exception
        finally:
            logger.shutdown_multiprocessing_logger()
        return result
    return wraps(function)(_logging)


def remove_test_logs(log_dir_path=LOG_PATH):
    """This method removes test logs"""
    paths = list(glob.glob(log_dir_path+'*.log'))
    for path in paths:
        os.remove(path)


def get_functions_with_prefix(prefix, obj):
    """
    This function return all functions with a prefix.
    :param prefix: string
    :param obj: object
                Object to investigate
    :return list of function objects
    """
    return [getattr(obj, func) for func in dir(obj) if func.startswith(prefix)]
