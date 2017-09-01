import abc
import time
import logging
import logging.handlers


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
        self.file_handler = None
        # This must be set at run
        self.result_logger = None

        # will be set in execute
        self.measured_time = None

        self.queue = None

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



    def execute(self, queue, logger_name):
        """
        Executes the experiment at hand by
        (1) calling run() and measuring the run time of run() and
        (2) calling analyze().
        """
        if self.queue is None:
            self.queue = queue

        if self.result_logger is None and self.file_handler is None:
            self.result_logger = setup_result_logger(queue, logger_name)
            self.file_handler = logging.FileHandler('%s.log' % self.log_name, mode='w')
            self.file_handler.setLevel(logging.DEBUG)
            self.progress_logger.addHandler(self.file_handler)

        start_time = time.time()
        self.run()
        self.measured_time = time.time() - start_time
        self.analyze()

def setup_result_logger(queue, logger_name):
    """
    This method setups a connection to the experimenter result logger.
    :param queue: Multiprocessing safe queue which is used to serialize the logging
    :param logger_name: Name of the experimenter result logger
    :return:
    """
    handler = logging.handlers.QueueHandler(queue)
    root = logging.getLogger(logger_name)
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    return root
