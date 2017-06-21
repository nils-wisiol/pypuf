import multiprocessing
import logging
from pypuf.tools import setup_logger


class Experimenter(object):
    """
        This class can be used to execute a list of experiments on several processors simultaneously.
    """
    # This static variable holds the message which signals the output thread that all processes are finished.
    FINISHED = 'EXPERIMENTER FINISHED'

    def __init__(self, log_name, experiments):
        """
        :param experiments: A list of pypuf.experiments.experiment.base.Experiment
        :param log_name: A unique file path where to output should be logged.
        """
        self.experiments = experiments
        self.log_name = log_name
        setup_logger(self.log_name, self.log_name)
        self.logger = logging.getLogger(self.log_name)
        self._cpu_count = multiprocessing.cpu_count()
        self.semaphore = multiprocessing.BoundedSemaphore(self._cpu_count)
        self.out_queue = multiprocessing.Queue()

    def gather_output(self, out_queue):
        """
            This method is used to gather all the outputs from experiments.
        """
        msg = out_queue.get()
        # log the experiment output
        while msg != Experimenter.FINISHED or out_queue.qsize() > 0:
            # log the output
            self.logger.info(msg+'\n')
            msg = out_queue.get()

    def run(self):
        """
            This method starts the logging and the execution of experiments.
        """
        # logging process
        logging_process = multiprocessing.Process(target=self.gather_output, args=(self.out_queue,))
        logging_process.start()

        jobs = []

        # start experiment jobs
        for exp in self.experiments:
            job = multiprocessing.Process(target=exp.execute, args=(self.out_queue, self.semaphore,))
            # is required to cap the amount of running experiments
            self.semaphore.acquire()
            job.start()
            jobs.append(job)

        # wait for all processes to be finished
        for job in jobs:
            job.join()

        # signal the output thread to be done
        self.out_queue.put(Experimenter.FINISHED)

        # wait for the master process to be done
        logging_process.join()
