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


class NoResultException(Exception):
    """Indicates that an experiment did not yield a result."""

    def __init__(self, experiment_id=None, experiment_pid=None, experiment_class=None):
        super().__init__()
        self.experiment_id = experiment_id
        self.experiment_pid = experiment_pid
        self.experiment_class = experiment_class


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
        self.id = uuid4()
        self.progress_log_name = progress_log_name
        self.parameters = parameters
        self.hash = sha256((self.__class__.__name__ + ': ' + str(parameters)).encode()).hexdigest()
        self.result = None

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

    def execute(self, result_log_queue, result_log_name):
        """
        Executes the experiment at hand by
        (1) calling run() and measuring the run time of run(),
        (2) calling analyze(),
        (3) logging the result
        :param result_log_queue: multiprocessing.queue
                      Multiprocessing safe queue which is used to coordinate the logging
        :param result_log_name: string
                        Name of the experimenter progress logger
        """
        try:
            # set up the progress logger
            file_handler = None
            if self.progress_log_name:
                self.progress_logger = logging.getLogger(self.progress_log_name)
                self.progress_logger.setLevel(logging.DEBUG)
                file_handler = logging.FileHandler('logs/%s.log' % self.progress_log_name, mode='w')
                file_handler.setLevel(logging.DEBUG)
                self.progress_logger.addHandler(file_handler)
                self.progress_logger.propagate = False

            # set up the result logger
            queue_handler = None
            self.result_logger = logging.getLogger(result_log_name)
            self.result_logger.setLevel(logging.DEBUG)
            if result_log_queue:
                queue_handler = logging.handlers.QueueHandler(result_log_queue)
                queue_handler.setLevel(logging.DEBUG)
                self.result_logger.addHandler(queue_handler)

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

            # clean up and return
            if self.progress_logger and file_handler:
                self.progress_logger.removeHandler(file_handler)
                file_handler.close()
            if self.result_logger and queue_handler:
                self.result_logger.removeHandler(queue_handler)
                queue_handler.close()

            return self.result
        except Exception as ex:
            # If anything goes wrong, we attach the experiment id for identification
            ex.experiment_id = self.id
            ex.pid = getpid()
            raise ex
