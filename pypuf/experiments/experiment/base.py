import abc
import time
import logging
from pypuf.tools import setup_logger


class Experiment(object):
    """
        This class is the inteface for all experiments executed by the pypuf.experiments.experimenter.Experimenter class
        and ensures compatibility.
    """

    def __init__(self, log_name, learner):
        """
        :param learner: A pypuf.simulation.base.Simulation instance
        :param log_name: A unique name, used for log path.
        """
        # This must be a valid file name to be able to log the results
        self.log_name = log_name
        setup_logger(self.log_name, self.log_name, write_console=False)
        self.logger = logging.getLogger(self.log_name)

        # The learner must be set to a valid parameter
        self.learner = learner
        # assign instance for analysing
        self.instance = self.learner.training_set.instance
        # must be set in execute
        self.model = None
        # must be set in execute
        self.start_time = 0
        # must be set in execute
        self.measured_time = 0

    @abc.abstractmethod
    def name(self):
        raise NotImplementedError('users must define output to use this base class')

    @abc.abstractmethod
    def output_string(self):
        raise NotImplementedError('users must define output to use this base class')

    def _output(self, queue):
        """
            This method should be used to pass the experiment result as string back to the master.
            This is necessary because the master process(Experimenter) can not get a deep copy of the experiment back.
        """
        msg = '{0}\n{1}'.format(self.name(), self.output_string())
        queue.put_nowait(msg)
        self.logger.info(msg)

    @abc.abstractmethod
    def analysis(self):
        """
            This method should analyse the results from learning which vary by the method of learning.
        """
        raise NotImplementedError('users must define analysis to use this base class')

    def execute(self, queue, semaphore):
        """
            This method is used by pypuf.experiments.experimenter.Experimenter. Every single spawned thread uses this
            method in oder to learn a trainingset. This method also handles the semaphore release call to coordinate the
            threads. If you want to provide iterative learning maybe with an certain stop criterion you have to override
            this method. If you override this method make sure to call semaphore().release() as last method in order to
            release resources.
        """
        self.start_time = time.time()
        self.model = self.learner.learn()
        self.measured_time = time.time() - self.start_time
        self.analysis()
        self._output(queue)
        semaphore.release()
