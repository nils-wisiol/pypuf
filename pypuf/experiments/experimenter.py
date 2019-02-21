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
import random
import sys
import traceback
import os
from datetime import datetime, timedelta
from threading import Timer, Lock


class Experimenter(object):
    """
    Coordinated, parallel execution of Experiments with logging.
    """

    def __init__(self, result_log_name, cpu_limit=2 ** 16, auto_multiprocessing=False, update_callback=None,
                 update_callback_min_pause=0):
        """
        :param result_log_name: A unique file path where to output should be logged.
        :param cpu_limit: Maximum number of parallel processes that run experiments.
        :param auto_multiprocessing: Whether to use numpy automatic multithreading/multiprocessing. Defaults to False.
        :param update_callback: If set, will be called every time an experiment finishes, see also
                update_callback_min_pause.
        :param update_callback_min_pause: If set, update_callbacks will be delayed to at least the number of seconds
                given here. If another experiment finishes during the pause, the earlier callback will be canceled.
        """

        # Store experiments list
        self.experiments = {}

        # Store logger name
        self.result_log_name = result_log_name

        # Setup parallel execution limit
        self.cpu_limit = min(cpu_limit, multiprocessing.cpu_count())
        self.semaphore = None

        # experimental results
        self.results = {}

        # Disable automatic multiprocessing
        if not auto_multiprocessing:
            self.disable_auto_multiprocessing()

        # Callback for new results
        self.update_callback = update_callback if update_callback else lambda *args: None
        self.update_callback_min_pause = update_callback_min_pause
        self.next_callback = None
        self.last_callback = None

        # Counters
        self.jobs_finished = 0
        self.jobs_total = 0
        self.jobs_errored = 0

    def queue(self, experiment):
        """
        Add an experiment to the queue.
        """
        self.experiments[experiment.id] = experiment
        self.results[experiment.id] = None
        self.jobs_total = len(self.experiments)
        return experiment.id

    def run(self, shuffle=False):
        """
        Runs all experiments. Blocks until all experiment are finished.
        """

        # Setup multiprocessing logging
        result_log_queue = multiprocessing.Manager().Queue()
        listener = multiprocessing.Process(target=result_log_listener,
                                           args=(result_log_queue, setup_result_logger, self.result_log_name,))
        listener.start()

        # Setup callback throttling
        result_lock = Lock()

        def call_callback(experiment_id=None, print_status=False):
            with result_lock:
                if print_status:
                    output_status(prefix='C')
                self.last_callback = datetime.now()
                self.update_callback(experiment_id)

        self.next_callback = Timer(0, call_callback)
        self.next_callback.start()

        # setup process pool
        self.jobs_total = len(self.experiments)
        start_time = datetime.now()
        print("Using up to %i CPUs with numpy multi-threading disabled" % self.cpu_limit)
        with multiprocessing.Pool(self.cpu_limit, maxtasksperchild=1) as pool:

            # print status function
            def output_status(prefix='F'):
                progress = self.jobs_finished / self.jobs_total
                elapsed_time = datetime.now() - start_time
                errors = "" if self.jobs_errored == 0 else " %i ERRORED, " % self.jobs_errored
                sys.stdout.write(
                    ("%s %s: %i jobs, %i finished, %i queued," + errors + " %.0f%%, ~remaining: %s\n") %
                    (
                        prefix,
                        datetime.now().strftime('%c'),
                        self.jobs_total,
                        self.jobs_finished,
                        self.jobs_total - self.jobs_finished,
                        progress * 100,
                        timedelta(seconds=(elapsed_time * (
                            1 - progress) / progress).total_seconds() // 15 * 15) if progress > 0 else '???',
                    )
                )

            # define callbacks, they are run within the main process, but in separate threads
            def update_status(result):
                self.jobs_finished += 1
                with result_lock:
                    self.results[result.experiment_id] = result
                output_status()

                # If there is already a callback waiting, we will replace it and therefore cancel it
                if self.next_callback.is_alive():
                    self.next_callback.cancel()

                # Schedule callback either immediately (0) or after the pause expired
                pause = max(0.0, self.update_callback_min_pause - (datetime.now() - self.last_callback).total_seconds())
                self.next_callback = Timer(
                    pause,
                    call_callback,
                    args=[result.experiment_id, pause != 0.0],
                )
                self.next_callback.start()

            def update_status_error(exception):
                print('Experiment exception: ', exception, file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                self.jobs_errored += 1
                update_status(exception)

            # randomize order
            experiments = list(self.experiments.values())
            if shuffle:
                random.seed(0xdeadbeef)
                random.shuffle(experiments)

            # experiment execution
            for experiment in experiments:
                pool.apply_async(
                    experiment.execute,
                    (result_log_queue, self.result_log_name),
                    callback=update_status,
                    error_callback=update_status_error,
                )

            # show status, then block until we're ready
            output_status()
            pool.close()
            pool.join()

            # quit logger
            result_log_queue.put(None)  # trigger listener to quit
            listener.join()

            # check if we got any exceptions as results
            exceptions = [ex for ex in self.results.values() if isinstance(ex, Exception)]
            if exceptions:
                raise FailedExperimentsException(exceptions)

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


def setup_result_logger(result_log_name):
    """
    This method is used to open the file handler for a logger_name. The resulting log file will have the format
    'logger_namer.log'.
    :param result_log_name: string
                        Path to or name to the log file
    """
    root = logging.getLogger(result_log_name)

    # Setup logging to both file and console
    file_handler = logging.FileHandler(filename='logs/%s.log' % result_log_name, mode='w')
    file_handler.setLevel(logging.INFO)

    root.addHandler(file_handler)
    return file_handler


def result_log_listener(queue, configurer, logger_name):
    """
    This is the root function of logging process which is responsible to log the experiment results.
    :param queue: multiprocessing.queue
                  This is used to coordinate the processes in order to obtain results.
    :param configurer: A function which setup the logger which logs the messages read from the queue.
    :param logger_name: String
                        Path to or name to the log file
    """
    handler = configurer(logger_name)
    while True:
        try:
            record = queue.get()
            if record is None:  # We send this as a sentinel to tell the listener to quit.
                handler.close()
                logging.getLogger(logger_name).removeHandler(handler)
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)  # No level or filter logic applied - just do it!
        except Exception as ex:  # pylint: disable=W
            print('Whoops! Problem:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            raise ex


class FailedExperimentsException(Exception):
    """
    Indicates that the experimenter's run included failed experiments.
    """
