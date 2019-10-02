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
import multiprocessing
import os
import random
import signal
import socket
import sys
import time
import traceback
from _sha256 import sha256
from datetime import datetime, timedelta
from multiprocessing.managers import SyncManager
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
        self.cpu_limit = int(os.environ.get('PYPUF_CPU_LIMIT', self.cpu_limit))
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
        if results_file:
            try:
                open('results/' + self.results_file).close()
            except FileNotFoundError:
                self.results.to_csv('results/' + self.results_file)
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
        print("Using up to %i CPUs %s" %
              (self.cpu_limit,
               'with numpy multi-threading disabled' if os.environ.get('OMP_NUM_THREADS', None) == '1' else ''))
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
                    print('Results file %s contains %i results to unknown experiments.' %
                          (self.results_file, len(unknown_experiments)))

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
        if os.environ.get('PYPUF_CPU_LIMIT', None) == '1':
            return

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
                                    'following environment: %s or restrict the number of '
                                    'parallel jobs that pypuf will run using the '
                                    'PYPUF_CPU_LIMIT environment variable set to 1.' % str(desired_environment))
        for key, val in desired_environment.items():
            os.environ[key] = val

    @staticmethod
    def _merge_results(results, other_results):
        """
        Returns a DataFrame containing all results of both arguments. If a result appears in
        both `results` and `other_results`, the result of `results` has precedence.
        """
        if len(other_results):
            merge = results.copy()
            return merge.append(other_results[~other_results.experiment_hash.isin(results.experiment_hash)])
        else:
            return results

    @property
    def _lock_file(self):
        return f'results/{self.results_file}.lock'

    @staticmethod
    def _lock_digest(my_id):
        return sha256(my_id.encode()).hexdigest()[:6]

    @property
    def _lock_id(self):
        my_id = f'{socket.getfqdn()}#{os.getpid()}'
        return f'{my_id}#{self._lock_digest(my_id)}'

    @property
    def _lock_owner_valid(self):
        owner = self._result_lock_owner
        if owner is None:
            return True
        parts = owner.split('#')
        if len(parts) != 3:
            return False
        owner_id = '#'.join(parts[0:2])
        digest = self._lock_digest(owner_id)
        return digest == parts[2]

    @property
    def _has_foreign_result_file_lock(self):
        owner = self._result_lock_owner
        return owner and owner != self._lock_id

    @property
    def _result_lock_owner(self):
        try:
            with open(self._lock_file) as f:
                return f.read()
        except OSError:
            return None

    def _acquire_result_file_lock(self, check_interval_s=.2, lock_timeout_s=900):
        try:
            while True:
                # wait for foreign process to give up the lock
                print(f'{time.time()} waiting for lock file for {self._lock_id} ...')
                while self._has_foreign_result_file_lock:
                    try:
                        wait_time = time.time() - os.path.getctime(self._lock_file)
                        if wait_time > lock_timeout_s:
                            # lock timeout, we forcibly take the lock to prevent data loss
                            self._release_result_file_lock(force=True)
                            print(f'{time.time()} Warning: Removing stuck lock file for owner '
                                  f'{self._has_foreign_result_file_lock} after timeout.')
                            break
                    except OSError:
                        pass

                    # wait and try again
                    sleep(abs(random.gauss(check_interval_s, .1)))

                # try to acquire the lock
                print(f'{time.time()} acquire lock file for {self._lock_id} ...')
                with open(self._lock_file, 'w') as f:
                    f.write(self._lock_id)

                # wait to see if someone else also tries to acquire the lock,
                # then check again if it is us now
                sleep(random.choice(range(4)))

                # break if we can confirm we have the lock
                if self._result_lock_owner == self._lock_id:
                    print(f'{time.time()} successfully acquired lock for {self._lock_file}')
                    break
                else:
                    if self._lock_owner_valid:
                        print(f'{time.time()} failed to acquire, lock owner is now {self._result_lock_owner}')
                    else:
                        print(f'{time.time()} failed to acquire, lock invalid: {self._result_lock_owner}')
                        self._release_result_file_lock(force=True)

            return True
        except OSError as e:
            print(f'{time.time()} Could not acquire result file lock: {e}')
            return False

    def _release_result_file_lock(self, force=False):
        if not self._has_foreign_result_file_lock or force:
            print(f'{time.time()} removing lock file')
            try:
                os.remove(self._lock_file)
            except OSError as e:
                print(f'{time.time()} error removing lock file: {e}')
                pass
        else:
            print(f'{time.time()} not removing foreign lock file, as force=False')

    def save_results(self):
        """
        Save current results to `results_file`.
        """
        if not self.results_file:
            return

        try:
            self._acquire_result_file_lock()

            from pandas import read_csv
            other_results = read_csv('results/' + self.results_file)

            self._merge_results(
                self.results,
                other_results,
            ).to_csv('results/' + self.results_file, index=False)
        finally:
            self._release_result_file_lock()

    def load_results(self):
        """
        Try to read results from `results_file`. Does nothing if that file does not exist.
        """
        from pandas.errors import EmptyDataError
        if self.results_file:
            try:
                self._acquire_result_file_lock()
                from pandas import read_csv
                self.results = read_csv('results/' + self.results_file)
            except (FileNotFoundError, EmptyDataError):
                pass
            finally:
                self._release_result_file_lock()


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
