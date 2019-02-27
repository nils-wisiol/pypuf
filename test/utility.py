"""
This module provides some utility functions in oder to minimize duplicate code in tests.
"""
import os
import glob
import multiprocessing
import sys
from io import StringIO
from functools import wraps
from pypuf.experiments.experimenter import result_log_listener, setup_result_logger

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
            target=result_log_listener,
            args=(self.queue, setup_result_logger, self.logger_name,)
        )

        self.listener.start()

    def shutdown_multiprocessing_logger(self):
        """This method shutdown the multiprocessing logging"""
        if self.listener.exitcode is None:
            self.queue.put_nowait(None)
            self.listener.join()

    def read_result_log(self):
        """
        This function is used to read from the result log. For this purpose the logger process have to be stopped.
        :return: string
                 Content of the result log.
        """
        self.shutdown_multiprocessing_logger()
        result_log = open('logs/' + self.logger_name+'.log', 'r')
        result = result_log.read()
        result_log.close()
        return result


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
    paths = list(glob.glob('logs/' + log_dir_path+'*.log'))
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
    return [func for func in dir(obj) if func.startswith(prefix)]


class _RedirectStream:
    """
    Context manager for temporarily redirecting a stream to another file.
    Copied from py35 contextlib because _RedirectStream is not available in py34 contextlib.
    """
    _stream = None

    def __init__(self, new_target):
        self._new_target = new_target
        # We use a list of old targets to make this CM re-entrant
        self._old_targets = []

    def __enter__(self):
        self._old_targets.append(getattr(sys, self._stream))
        setattr(sys, self._stream, self._new_target)
        return self._new_target

    def __exit__(self, exctype, excinst, exctb):
        setattr(sys, self._stream, self._old_targets.pop())


class RedirectStdout(_RedirectStream):
    """
    Context manager for temporarily redirecting stdout to another file.
    Copied from py35 contextlib because RedirectStdout is not available in py34 contextlib.
    """

    _stream = "stdout"


class RedirectStderr(_RedirectStream):
    """
    Context manager for temporarily redirecting stderr to another file.
    Copied from py35 contextlib because RedirectStderr is not available in py34 contextlib.
    """

    _stream = "stderr"


def mute(function):
    """
    This function provides an function decorator which redirects the stderr and stdout stream to an stream object
    in order to prevent console output.
    """
    def _mute(*args, **kwargs):
        """
        This method absorbs stderr and stdout streams.
        :param args: extra positional generic arguments
        :param kwargs: extra keyword arguments generic argument
        :return: return value of `function`
        """
        output_sink = StringIO()
        try:
            from contextlib import redirect_stdout, redirect_stderr
            redirect_out = redirect_stdout(output_sink)
            redirect_err = redirect_stderr(output_sink)
        except ImportError:
            redirect_out = RedirectStdout(output_sink)
            redirect_err = RedirectStderr(output_sink)
        try:
            # with redirect_stdout(output_sink):
            with redirect_err, redirect_out:
                result = function(*args, **kwargs)
        except Exception as exception:
            raise exception
        return result
    return wraps(function)(_mute)
