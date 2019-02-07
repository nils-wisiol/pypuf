"""
This module provides an abstract class which can be used by a pypuf.experiments.experimenter.Experimenter object in
order to be executed.
"""
import abc
import logging
import logging.handlers
import sys
from time import time, clock


class Experiment(object):
    """
    This class defines an experiment, mainly consisting of instructions how to run and analyze it (methods run() and
    analyze(), respectively). It can be used with the Experimenter class to run Experiments in parallel.
    """

    def __init__(self, log_name):
        """
        :param log_name: A unique name, used for log path.
        """

        # Setup logger for experiment specific logging
        self.progress_logger = logging.getLogger(log_name)
        self.log_name = log_name
        self.progress_logger.setLevel(logging.DEBUG)

        # This must be set at run
        self.result_logger = None

        # will be set in execute
        self.measured_time = None

    @abc.abstractmethod
    def analyze(self):
        """
        This method analyzes the results of the experiment.
        """
        raise NotImplementedError('users must define analysis to use this base class')

    def prepare(self):
        """
        Used for preparation work that shall be not timed.
        Executed just before run()
        """
        pass

    @abc.abstractmethod
    def run(self):
        """
        This method runs the actual experiment.
        """
        raise NotImplementedError('users must define run() to use this base class')

    def execute(self, queue, logger_name):
        """
        Executes the experiment at hand by
        (1) calling run() and measuring the run time of run() and
        (2) calling analyze().
        :param queue: multiprocessing.queue
                      Multiprocessing safe queue which is used to serialize the logging
        :param logger_name: string
                        Name of the experimenter result logger
        """
        self.result_logger = setup_result_logger(queue, logger_name)
        file_handler = logging.FileHandler('logs/%s.log' % self.log_name, mode='w')
        file_handler.setLevel(logging.DEBUG)
        self.progress_logger.addHandler(file_handler)
        self.prepare()

        if sys.platform == 'win32':
            timer = clock
        else:
            timer = time

        start_time = timer()
        self.run()
        self.measured_time = timer() - start_time
        self.analyze()


def setup_result_logger(queue, logger_name):
    """
    This method setups a connection to the experimenter result logger.
    :param queue: multiprocessing.queue
                  Multiprocessing safe queue which is used to serialize the logging
    :param logger_name: string
                        Name of the experimenter result logger
    """
    handler = logging.handlers.QueueHandler(queue)
    root = logging.getLogger(logger_name)
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    return root
