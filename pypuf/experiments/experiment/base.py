import abc
import time
import logging


class Experiment(object):
    """
    This class defines an experiment, mainly consisting of instructions how to run and analyze it (methods run() and
    analyze(), respectively). It can be used with the Experimenter class to run Experiments in parallel.
    """

    def __init__(self, log_name):
        """
        :param log_name: A unique name, used for log path.
        """

        # This must be a valid file name to be able to log the results
        self.progress_logger = logging.getLogger(log_name)
        file_handler = logging.FileHandler('%s.log' % log_name, mode='w')
        file_handler.setLevel(logging.DEBUG)
        self.progress_logger.addHandler(file_handler)
        self.progress_logger.setLevel(logging.DEBUG)
        self.result_logger = self.progress_logger.parent

        # will be set in execute
        self.measured_time = None

    @abc.abstractmethod
    def analyze(self):
        """
        This method analyzes the results of the experiment.
        """
        raise NotImplementedError('users must define analysis to use this base class')

    @abc.abstractmethod
    def run(self):
        """
        This method runs the actual experiment.
        """
        raise NotImplementedError('users must define run() to use this base class')

    def execute(self):
        """
        Executes the experiment at hand by
        (1) calling run() and measuring the run time of run() and
        (2) calling analyze().
        """
        start_time = time.time()
        self.run()
        self.measured_time = time.time() - start_time
        self.analyze()
