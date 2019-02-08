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
import os
from datetime import datetime, timedelta


class Experimenter(object):
    """
    Coordinated, parallel execution of Experiments with logging.
    """

    def __init__(self, log_name, experiments, cpu_limit=2**16, auto_multiprocessing=False, update_callback=None):
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
        self.semaphore = None

        # experimental results
        self.results = []

        # Disable automatic multiprocessing
        if not auto_multiprocessing:
            self.disable_auto_multiprocessing()

        # Callback for new results
        self.update_callback = update_callback if update_callback else lambda *args: None

        # Counters
        self.jobs_finished = 0
        self.jobs_total = len(self.experiments)

    def run(self):
        """
        Runs all experiments. Blocks until all experiment are finished.
        """

        # Setup multiprocessing logging
        logging_queue = multiprocessing.Manager().Queue(-1)
        listener = multiprocessing.Process(target=log_listener,
                                           args=(logging_queue, setup_logger, self.logger_name,))
        listener.start()

        # setup process pool
        self.jobs_total = len(self.experiments)
        start_time = datetime.now()
        print("Using up to %i CPUs with numpy multi-threading disabled" % self.cpu_limit)
        with multiprocessing.Pool(self.cpu_limit, maxtasksperchild=1) as pool:

            # print status function
            def output_status():
                progress = self.jobs_finished / self.jobs_total
                elapsed_time = datetime.now() - start_time
                sys.stdout.write(
                    "%s: %i jobs total, %i finished, %i queued, %.2f, ~remaining: %s\n" %
                    (
                        datetime.now().strftime('%c'),
                        self.jobs_total,
                        self.jobs_finished,
                        self.jobs_total - self.jobs_finished,
                        progress,
                        timedelta(seconds=(elapsed_time * (
                                    1 - progress) / progress).total_seconds() // 15 * 15) if progress > 0 else '???',
                    )
                )
                self.update_callback()

            # define callbacks, they are run within the main process, but in separate threads
            def update_status(result):
                self.jobs_finished += 1
                self.results.append(result)
                output_status()

            def update_status_error(exception):
                raise exception

            # experiment execution
            for i, experiment in enumerate(self.experiments):
                pool.apply_async(
                    experiment.execute,
                    (logging_queue, self.logger_name),
                    callback=update_status,
                    error_callback=update_status_error,
                )

            # block until we're ready
            pool.close()
            pool.join()

            # quit logger
            logging_queue.put_nowait(None)

    @staticmethod
    def disable_auto_multiprocessing():
        """
        Disables numpy's automatic multiprocessing/multithreading for the current
        python instance by setting environment variables.
        This method must be called before numpy is first imported. If numpy was
        already imported and the environment was not yet set accordingly, an
        Exception will be raised.
        """
        desired_environment = {
            'OMP_NUM_THREADS': '1',
            'NUMEXPR_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
        }
        if 'numpy' in sys.modules:
            for key, val in desired_environment.items():
                if key not in os.environ or os.environ[key] != val:
                    raise Exception('Cannot disable numpy\'s automatic parallel computing, '
                                    'because it was already imported. To fix this issue, '
                                    'import Experimenter before numpy or run python in the '
                                    'following environment: ' + str(desired_environment))
        for key, val in desired_environment.items():
            os.environ[key] = val


def setup_logger(logger_name):
    """
    This method is used to open the file handler for a logger_name. The resulting log file will have the format
    'logger_namer.log'.
    :param logger_name: string
                        Path to or name to the log file
    """
    root = logging.getLogger(logger_name)

    # Setup logging to both file and console
    file_handler = logging.FileHandler(filename='logs/%s.log' % logger_name, mode='w')
    file_handler.setLevel(logging.INFO)

    root.addHandler(file_handler)


def log_listener(queue, configurer, logger_name):
    """
    This is the root function of logging process which is responsible to log the experiment results.
    :param queue: multiprocessing.queue
                  This is used to coordinate the processes in order to obtain results.
    :param configurer: A function which setup the logger which logs the messages read from the queue.
    :param logger_name: String
                        Path to or name to the log file
    """
    configurer(logger_name)
    while True:
        try:
            record = queue.get()
            if record is None:  # We send this as a sentinel to tell the listener to quit.
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)  # No level or filter logic applied - just do it!
        except Exception:  # pylint: disable=W
            print('Whoops! Problem:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
