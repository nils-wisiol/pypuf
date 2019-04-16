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
import logging
import os
import random
import signal
import sys
import traceback
import multiprocessing
from multiprocessing.managers import SyncManager
from datetime import datetime, timedelta
from threading import Timer, Lock
from time import sleep

from pypuf.experiments.experiment.base import ExperimentCanceledException


class Experimenter(object):
    """
    Coordinated, parallel execution of Experiments with logging.
    """

    def __init__(self, result_log_name, cpu_limit=None, gpu_limit=None,
                 auto_multiprocessing=False, update_callback=None,
                 update_callback_min_pause=0, results_file=None):
        """
        :param result_log_name: A unique file path where to output should be logged.
        :param cpu_limit: Maximum number of parallel processes that run experiments.
        :param gpu_limit: Number of GPUs that are available on the system and can be used for experiments
                (1 GPU per experiment/process).
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
        self.cpu_limit = min(cpu_limit, multiprocessing.cpu_count()) if cpu_limit else multiprocessing.cpu_count()
        self.semaphore = None

        # Setup GPU usage
        self.gpu_limit = gpu_limit
        self.gpu_counter = 0
        if self.gpu_limit is None:
            self.gpu_limit = 0

        # experimental results
        from pandas import DataFrame
        self.results = DataFrame()
        self.exceptions = []
        self.results_file = results_file
        self.load_results()

        # Disable automatic multiprocessing
        if not auto_multiprocessing:
            self.disable_auto_multiprocessing()

        # Callback for new results
        self.update_callback = update_callback if update_callback else lambda *args: None
        self.update_callback_min_pause = update_callback_min_pause
        self.next_callback = None
        self.last_callback = None
        self.callback_disabled = False

        # Counters
        self.jobs_finished = 0
        self.jobs_total = 0
        self.jobs_errored = 0

        # Interrupt Handling
        self.num_int = 0
        self.cancel_experiments = None
        self.interrupt_condition = None

    def queue(self, experiment):
        """
        Add an experiment to the queue.
        """
        self.experiments[experiment.id] = experiment
        self.jobs_total = len(self.experiments)
        return experiment.hash

    def run(self, shuffle=False):
        """
        Runs all experiments. Blocks until all experiment are finished.
        """

        # Setup multiprocessing logging
        manager = SyncManager()
        manager.start(lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))
        result_log_queue = manager.Queue()
        self.cancel_experiments = manager.Value('b', 0)
        self.interrupt_condition = manager.Condition()
        listener = multiprocessing.Process(target=result_log_listener,
                                           args=(result_log_queue, setup_result_logger, self.result_log_name,))
        listener.start()

        # Setup callback throttling
        result_lock = Lock()
        callback_lock = Lock()
        self.callback_disabled = False

        def call_callback(experiment_id=None, pause=0):
            with callback_lock:
                with result_lock:
                    if pause:
                        output_status(prefix='C')
                    self.last_callback = datetime.now()
                    self.save_results()
                    sys.stdout.write('Results saved. ')
                    sys.stdout.flush()
                    if not self.callback_disabled:
                        sys.stdout.write('Digesting results ... ')
                        sys.stdout.flush()

                        try:
                            self.update_callback(experiment_id)
                        except Exception as ex:  # pylint: disable=W
                            sys.stdout.write('errored with {}\n\n\n'.format(ex.__class__))
                            self.callback_disabled = True
                        else:
                            sys.stdout.write('done\n')
                    else:
                        sys.stdout.write('Digestion disabled, due to previous exception.\n')

        self.last_callback = datetime.now()

        # setup process pool
        self.jobs_total = len(self.experiments)
        start_time = datetime.now()
        print("Using up to %i CPUs with numpy multi-threading disabled" % self.cpu_limit)
        with multiprocessing.Pool(self.cpu_limit) as pool:

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
                        timedelta(seconds=(elapsed_time * (1 - progress) / progress).total_seconds() // 15 * 15)
                        if progress > 0 else '???',
                    )
                )

            # define callbacks, they are run within the main process, but in separate threads
            def update_status(result=None):
                from pandas import DataFrame
                with result_lock:
                    self.jobs_finished += 1
                    output_status()
                    if not result:
                        return

                    row = {}
                    experiment = self.experiments[result.experiment_id]
                    row.update({
                        'experiment_id': result.experiment_id,
                        'experiment_hash': experiment.hash,
                        'experiment': experiment.__class__.__name__,
                    })
                    row.update(experiment.parameters._asdict())
                    row.update(result._asdict())
                    self.results = self.results.append(DataFrame([row]), sort=True)

                # If there is already a callback waiting, we will replace it and therefore cancel it
                if self.next_callback and self.next_callback.is_alive():
                    sleep(0)  # let other threads run first
                    if callback_lock.acquire(blocking=False):
                        self.next_callback.cancel()
                        callback_lock.release()
                    else:
                        # the callback is currently waiting for the result_lock
                        return

                # Schedule callback either immediately (0) or after the pause expired
                pause = max(0.0, self.update_callback_min_pause - (datetime.now() - self.last_callback).total_seconds())
                self.next_callback = Timer(
                    pause,
                    call_callback,
                    args=[result.experiment_id, pause],
                )
                self.next_callback.start()

            def update_status_error(exception):
                if isinstance(exception, ExperimentCanceledException):
                    return
                print('Experiment exception: ', exception, file=sys.stderr)
                traceback.print_exception(type(exception), exception, exception.__traceback__, file=sys.stderr)
                self.jobs_errored += 1
                self.exceptions.append(exception)
                update_status()

            # randomize order
            experiments = list(self.experiments.values())
            if shuffle:
                random.seed(0xdeadbeef)
                random.shuffle(experiments)

            # filter loaded experiments
            if not self.results.empty:
                known_hashes = [ex.hash for ex in experiments]
                len_before = len(experiments)
                loaded_experiment_hashes = self.results.loc[:, ['experiment_hash']].values[:, 0]
                experiments = [ex for ex in experiments if ex.hash not in loaded_experiment_hashes]
                if loaded_experiment_hashes.size:
                    print('Continuing from %s' % self.results_file)
                    self.jobs_finished = len_before - len(experiments)

                # check for experiments with results that we don't know
                unknown_experiments = self.results.loc[~self.results['experiment_hash'].isin(known_hashes)]
                if not unknown_experiments.empty:
                    print('@' * 80)
                    print('Results file %s contains %i results that are not in the study\'s' %
                          (self.results_file, len(unknown_experiments)))
                    print('experiment definition. Did you delete experiments from your study?')
                    print('@' * 80)

            # experiment execution
            for experiment in experiments:
                # Assign experiment to GPU (if used) : might be replaced by more sophisticated load balancer
                if self.gpu_limit > 0:
                    gpu_num = self.gpu_counter % self.gpu_limit
                    experiment.assign_to_gpu(gpu_num)
                    self.gpu_counter += 1
                # Add experiment to execution queue
                pool.apply_async(
                    experiment.execute,
                    (result_log_queue, self.result_log_name, self.cancel_experiments, self.interrupt_condition),
                    callback=update_status,
                    error_callback=update_status_error,
                )

            def signal_handler(_sig, _frame):
                self.num_int += 1
                if self.num_int > 1:
                    print("Killing all processes.")
                    sys.exit(1)
                print(
                    "\rPerforming graceful shutdown... (Press CTRL-C again to force. This might result in data loss.)")
                with self.interrupt_condition:
                    self.cancel_experiments.value = 1
                    self.interrupt_condition.notify_all()

            signal.signal(signal.SIGINT, signal_handler)

            # show status, then block until we're ready
            output_status()
            pool.close()

            pool.join()

            if self.next_callback and self.next_callback.is_alive():
                with callback_lock:
                    self.next_callback.cancel()

            call_callback()

            # quit logger
            result_log_queue.put(None)  # trigger listener to quit
            listener.join()

            # check if we got any exceptions as results
            if self.exceptions:
                raise FailedExperimentsException(self.exceptions)

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

    def save_results(self):
        """
        Save current results to `results_file`.
        """
        if not self.results_file:
            return
        self.results.to_csv('results/' + self.results_file, index=False)

    def load_results(self):
        """
        Try to read results from `results_file`. Does nothing if that file does not exist.
        """
        from pandas.errors import EmptyDataError
        try:
            if self.results_file:
                from pandas import read_csv
                self.results = read_csv('results/' + self.results_file)
        except (FileNotFoundError, EmptyDataError):
            pass


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
    signal.signal(signal.SIGINT, signal.SIG_IGN)
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
