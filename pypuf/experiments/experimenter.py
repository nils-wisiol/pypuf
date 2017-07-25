import multiprocessing
import logging


class Experimenter(object):
    """
    Coordinated, parallel execution of Experiments with logging.
    """

    def __init__(self, log_name, experiments, cpu_limit=2**16):
        """
        :param experiments: A list of pypuf.experiments.experiment.base.Experiment
        :param log_name: A unique file path where to output should be logged.
        :param cpu_limit: Maximum number of parallel processes that run experiments.
        """

        # Store experiments list
        self.experiments = experiments

        # Setup logging to both file and console
        file_handler = logging.FileHandler(filename='%s.log' % log_name, mode='w')
        file_handler.setLevel(logging.INFO)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)

        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

        # Setup parallel execution limit
        self.cpu_limit = min(cpu_limit, multiprocessing.cpu_count())
        self.semaphore = multiprocessing.BoundedSemaphore(self.cpu_limit)

    def run(self):
        """
        Runs all experiments.
        """

        jobs = []

        for exp in self.experiments:

            # define experiment process
            def run_experiment(semaphore):
                exp.execute()  # run the actual experiment
                semaphore.release()  # release CPU

            job = multiprocessing.Process(
                target=run_experiment,
                args=(self.semaphore,)
            )

            # run experiment
            self.semaphore.acquire()  # wait for a free CPU
            job.start()

            # keep a list of all jobs
            jobs.append(job)

        # wait for all processes to be finished
        for job in jobs:
            job.join()

