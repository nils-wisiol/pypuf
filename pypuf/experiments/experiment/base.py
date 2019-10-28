"""
This module provides an abstract class which can be used by a pypuf.experiments.experimenter.Experimenter object in
order to be executed.
"""
import abc
import logging
import logging.handlers
import sys
from _sha256 import sha256
from os import getpid
from time import time, clock
from uuid import uuid4

PROC_STATUS_MEMORY_INFO = ['VmPeak', 'VmSize', 'VmLck', 'VmPin', 'VmHWM', 'VmRSS', 'RssAnon', 'RssFile', 'RssShmem',
                           'VmData', 'VmStk', 'VmExe', 'VmLib', 'VmPTE', 'VmSwap', 'HugetlbPages']


def memory_info():
    """
    Obtain memory info about the current python process. Linux only.
    :return: Dictionary keyed with kernel memory information names and number of byte values.
    """
    def convert(proc_memory_size_value_kb: str):
        return int(proc_memory_size_value_kb.strip('kB \t')) * 1024

    # noinspection PyBroadException
    try:
        status = proc_status()
    except (OSError, LookupError, TypeError, EOFError, ValueError):
        return {}

    result = {}

    for key in PROC_STATUS_MEMORY_INFO:
        try:
            result[key] = convert(status[key][0])
        except LookupError:
            result[key] = None

    return result


def proc_status():
    """
    Read and process /proc/self/status.
    :return: Dictionary with information in /proc/self/status.
    """
    with open('/proc/self/status') as fh:
        s = fh.read()
    return {
        line.split('\t', 1)[0].rstrip(':'): line.split('\t')[1:]
        for line in s.split('\n')
        if line
    }


class NoResultException(Exception):
    """Indicates that an experiment did not yield a result."""

    def __init__(self, experiment_id=None, experiment_pid=None, experiment_class=None):
        super().__init__()
        self.experiment_id = experiment_id
        self.experiment_pid = experiment_pid
        self.experiment_class = experiment_class


class ExperimentCanceledException(Exception):
    """Indicates that an experiment has been canceled."""


class LogMemoryUsageLoggerAdapter(logging.LoggerAdapter):
    """Provides logger with memory usage (VmRSS) information, in Gigabytes."""
    propagate = False

    def process(self, msg, kwargs):
        try:
            self.extra['memory_gib'] = memory_info()['VmRSS'] / 1024**3
        except (TypeError, KeyError):
            self.extra['memory_gib'] = float('nan')
        return super().process(msg, kwargs)


class Experiment(object):
    """
    This class defines an experiment, mainly consisting of instructions how to run and analyze it (methods run() and
    analyze(), respectively). It can be used with the Experimenter class to run Experiments in parallel.
    """

    def __init__(self, progress_log_name, parameters):
        """
        :param progress_log_name: A unique name, used for log path.
        :param parameters: NamedTuple object holding all experiment parameters
        """
        self.id = getattr(self, 'id', uuid4())
        self.progress_log_name = progress_log_name
        self.parameters = parameters
        self.hash = sha256((self.__class__.__name__ + ': ' + str(parameters)).encode()).hexdigest()
        self.result = None

        # GPU is unset. Experimenter uses assign_to_gpu() before executing experiment
        self.gpu_id = None

        # This must be set at run, loggers can (under circumstances) not be pickled
        self.progress_logger = None
        self.result_logger = None

        # Prepare time measurement
        self.timer = clock if sys.platform == 'win32' else time
        self.measured_time = None

    @abc.abstractmethod
    def analyze(self):
        """
        This method analyzes the results of the experiment.
        :return Instance of ExperimentResult, holding all relevant results.
        """
        raise NotImplementedError('users must define analysis to use this base class')

    def prepare(self):
        """
        Used for preparation work that shall be not timed.
        Executed just before run()
        """

    @abc.abstractmethod
    def run(self):
        """
        This method runs the actual experiment.
        """
        raise NotImplementedError('users must define run() to use this base class')

    def assign_to_gpu(self, gpu_id):
        """
            Set gpu_id. Called by Experimenter to load-balance GPUs
        """
        self.gpu_id = gpu_id

    def execute(self, result_log_queue, result_log_name, cancel_experiment=None, interrupt_condition=None):
        """
        Executes the experiment at hand by
        (1) calling run() and measuring the run time of run(),
        (2) calling analyze(),
        (3) logging the result
        :param result_log_queue: multiprocessing.queue
                      Multiprocessing safe queue which is used to coordinate the logging
        :param result_log_name: string
                        Name of the experimenter progress logger
        :param cancel_experiment: multiprocessing.Value indicating if experiment is canceled
        :param interrupt_condition: multiprocessing.Condition to sync interrupt behavior
        """
        file_handler = None
        queue_handler = None
        try:
            if cancel_experiment and cancel_experiment.value == 1:
                raise ExperimentCanceledException()
            # set up the progress logger
            if self.progress_log_name:
                self.progress_logger = LogMemoryUsageLoggerAdapter(logging.getLogger(self.progress_log_name), {})
                self.progress_logger.setLevel(logging.DEBUG)
                file_handler = logging.FileHandler('logs/%s.log' % self.progress_log_name, mode='w')
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(
                    logging.Formatter(fmt='%(asctime)s %(memory_gib).2fGiB %(levelname)-8s %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
                )
                self.progress_logger.logger.addHandler(file_handler)
                self.progress_logger.propagate = False

            # set up the result logger
            self.result_logger = LogMemoryUsageLoggerAdapter(logging.getLogger(result_log_name), {})
            self.result_logger.setLevel(logging.DEBUG)
            if result_log_queue:
                queue_handler = logging.handlers.QueueHandler(result_log_queue)
                queue_handler.setLevel(logging.DEBUG)
                logging.getLogger(result_log_name).addHandler(queue_handler)

            # run preparations (not timed)
            self.prepare()

            # run the actual experiment
            start_time = self.timer()
            self.run()
            self.measured_time = self.timer() - start_time

            # analyze the result
            self.result = self.analyze()
            if not self.result:
                raise NoResultException(
                    experiment_id=self.id,
                    experiment_pid=getpid(),
                    experiment_class=self.__class__.__name__
                )
            self.result_logger.info(str(self.result).replace("\n", ''))

            return self.result
        except KeyboardInterrupt:
            with interrupt_condition:
                interrupt_condition.wait_for(lambda: cancel_experiment.value == 1)
            raise ExperimentCanceledException()
        except Exception as ex:
            # If anything goes wrong, we attach the experiment id for identification
            ex.experiment_id = self.id
            ex.pid = getpid()
            raise ex
        finally:
            # clean up and return
            if self.progress_logger and file_handler:
                self.progress_logger.logger.removeHandler(file_handler)
                file_handler.close()
            if self.result_logger and queue_handler:
                logging.getLogger(result_log_name).removeHandler(queue_handler)
                queue_handler.close()

    @staticmethod
    def max_memory():
        """
        Information about peak memory usage.
        :return: peak memory usage in bytes (int) or float('nan') if not available
        """
        try:
            return memory_info()['VmPeak']
        except KeyError:
            return float('nan')
