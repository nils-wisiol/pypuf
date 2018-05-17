"""
This module provides a class which acts as process pool. The is build in order to distribute several experiments over
the system cores. The results of the experiments are written into on a single log file which is coordinated by a logging
process. In order to run experiments a path to a logfile and list of experiments is needed which inherit the
pypuf.experiments.experiment.base class.

Example usage:

experiments = []
n = 128
for i in range(n):
    log_name = 'test_multiprocessing_logs{0}'.format(i)
    experiment = Experiment(...)
    experiments.append(experiment)

experimenter = Experimenter('experimenter_log_path', experiments)
experimenter.run()
"""
import multiprocessing
import logging
import sys
import traceback
import time
from numpy import mean


class Experimenter(object):
    """
    Coordinated, parallel execution of Experiments with logging.
    """

    PRINT_DELIMITER = 'PRINT:'

    def __init__(self, log_name, experiments, status_display=False, cpu_limit=2**16):
        """
        :param experiments: A list of pypuf.experiments.experiment.base.Experiment
        :param status_display: boolean
                               This value determines to display status information about the experiment execution.
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
        self.status_display = status_display

    def run(self):
        """
        Runs all experiments.
        """

        # Setup multiprocessing logging
        queue = multiprocessing.Queue(-1)
        listener = multiprocessing.Process(
            target=log_listener,
            args=(queue, setup_logger, self.logger_name, len(self.experiments), self.status_display,)
        )
        listener.start()

        # list of active jobs
        active_jobs = []

        for exp in self.experiments:

            # define experiment process
            def run_experiment(experiment, queue, semaphore, logger_name):
                """
                This method is responsible to start the experiment and release the semaphore which coordinates the
                number of parallel running processes.
                :param experiment: pypuf.experiments.experiment.base.Experiment
                                   A implementation of the base experiment class which should be executed
                :param queue: multiprocessing.queue
                              This is used to coordinate the processes in order to obtain results.
                :param semaphore: multiprocessing.BoundedSemaphore(self.cpu_limit)
                                  A semaphore to limit the number of concurrent running experiments
                :param logger_name: String
                                    Path to or name to the log file of the experiment
                """
                try:
                    experiment.execute(queue, logger_name)  # run the actual experiment
                    semaphore.release()  # release CPU
                except Exception as experiment:  # pylint: disable=W
                    semaphore.release()  # release CPU
                    raise experiment

            job = multiprocessing.Process(
                target=run_experiment,
                args=(exp, queue, self.semaphore, self.logger_name)
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
    """
    This method is used to open the file handler for a logger_name. The resulting log file will have the format
    'logger_namer.log'.
    :param logger_name: string
                        Path to or name to the log file
    """
    root = logging.getLogger(logger_name)

    # Setup logging to both file and console
    file_handler = logging.FileHandler(filename='%s.log' % logger_name, mode='w')
    file_handler.setLevel(logging.INFO)

    root.addHandler(file_handler)


def log_listener(queue, configurer, logger_name, max_exp=1, status_display=False):
    """
    This is the root function of logging process which is responsible to log the experiment results.
    :param queue: multiprocessing.queue
                  This is used to coordinate the processes in order to obtain results.
    :param configurer: A function which setup the logger which logs the messages read from the queue.
    :param logger_name: String
                        Path to or name to the log file
    :param max_exp: int default is 0
                    Number of experiments to execute.
    :param status_display: boolean
                           This value determines to display status information about the experiment execution.
    """
    start_time = time.time()
    configurer(logger_name)
    experiments_done = 0
    times = []
    runtimes = 0.0
    if status_display:
        # Print status header
        header = '\rFinished Experiments|Average Execution Time|Estimated Runtime|Runtime\n'
        sys.stderr.write(header)
        sys.stderr.flush()

    while True:
        if status_display:
            current_runtime = time.time() - start_time + runtimes
            # Refresh the status display
            print_status(times, experiments_done, max_exp, current_runtime)
        try:
            if queue.empty():
                continue
            if status_display:
                record = queue.get_nowait()
            else:
                record = queue.get()
            if record is None:  # We send this as a sentinel to tell the listener to quit.
                break

            # Recognize experiments end and get the execution time
            if record.msg.startswith(Experimenter.PRINT_DELIMITER):
                times.append(float(record.msg.replace(Experimenter.PRINT_DELIMITER, '')))
                runtimes += times[-1]
                experiments_done += 1
                continue
            else:
                # Log the result into a file
                logger = logging.getLogger(record.name)
                logger.handle(record)  # No level or filter logic applied - just do it!
        except Exception:  # pylint: disable=W
            print('Whoops! Problem:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


def print_status(times, experiments_done, max_experiments, runtime):
    """
    This function can be used to display information about the experimenter runtime.
    :param times: list of float
                  List of experiment execution times
    :param experiments_done: int
                             Number fo finished experiments.
    :param max_experiments: int
                            Number of experiments to execute
    :param runtime: float
                    Time the experimenter runs.
    """
    if times == []:
        sys.stderr.write('\rwait for finished experiments...')
        sys.stderr.flush()
        return

    mean_exe_time = round(mean(times), 2)
    estimated_runtime = round(mean_exe_time * max_experiments, 2)
    exp_overview = '{}/{}'.format(experiments_done, max_experiments)
    values = '\r{:<20}|{:<22}|{:<17}|{:<9}'.format(exp_overview, mean_exe_time, estimated_runtime, round(runtime, 2))
    sys.stderr.write(values)
    sys.stderr.flush()
