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

        # Store logger name
        self.logger_name = log_name

        # Setup parallel execution limit
        self.cpu_limit = min(cpu_limit, multiprocessing.cpu_count())
        self.semaphore = multiprocessing.BoundedSemaphore(self.cpu_limit)

    def run(self):
        """
        Runs all experiments.
        """

        # Setup multiprocessing logging
        queue = multiprocessing.Queue(-1)
        listener = multiprocessing.Process(target=log_listener,
                                           args=(queue, setup_logger, self.logger_name,))
        listener.start()

        # list of active jobs
        active_jobs = []

        for exp in self.experiments:

            # define experiment process
            def run_experiment(queue, semaphore, logger_name):
                try:
                    exp.execute(queue, logger_name)  # run the actual experiment
                    semaphore.release()  # release CPU
                except Exception as e:
                    semaphore.release()  # release CPU
                    raise e

            job = multiprocessing.Process(
                target=run_experiment,
                args=(queue, self.semaphore, self.logger_name)
            )

            # wait for a free CPU
            self.semaphore.acquire()

            def list_active_jobs():
                """
                update list of active jobs
                return: [multiprocessing.Process] list of active jobs
                """
                still_active_jobs = []
                for j in active_jobs:
                    j.join(0)
                    if j.exitcode is None:
                        still_active_jobs.append(j)
                return still_active_jobs

            # start experiment
            job.start()
            active_jobs = list_active_jobs()
            active_jobs.append(job)

        # wait for all processes to be finished
        for job in active_jobs:
            job.join()

        # Quit logging process
        queue.put_nowait(None)
        listener.join()


def setup_logger(logger_name):
    root = logging.getLogger(logger_name)

    # Setup logging to both file and console
    file_handler = logging.FileHandler(filename='%s.log' % logger_name, mode='w')
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    root.addHandler(file_handler)
    root.addHandler(stream_handler)


def log_listener(queue, configurer, logger_name):
    configurer(logger_name)
    while True:
        try:
            record = queue.get()
            if record is None:  # We send this as a sentinel to tell the listener to quit.
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)  # No level or filter logic applied - just do it!
        except Exception:
            import sys, traceback
            print('Whoops! Problem:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)