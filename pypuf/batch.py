import logging
import os
import pickle
import random
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import List, Optional

import pandas as pd
from numpy.distutils import cpuinfo
from memory_profiler import memory_usage

# noinspection PyBroadException
try:
    cpu = cpuinfo.cpu.info[0]['model name']
except Exception:
    cpu = None


class ResultCollection:

    def __init__(self) -> None:
        pass

    def add_result(self, parameter_hash: str, result: dict) -> None:
        raise NotImplementedError

    def known_results(self) -> List[str]:
        raise NotImplementedError

    def save_log(self, log: object, params: dict, parameter_hash: str, force: bool = False) -> None:
        raise NotImplementedError


class PickleResultCollection(ResultCollection):

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path
        self.log_path = self.path + '.log.pickle'
        self.log_saved = datetime.now()
        try:
            self.results = pd.read_pickle(self.path)
        except FileNotFoundError:
            self.results = pd.DataFrame()

    def add_result(self, parameter_hash: str, result: dict) -> None:
        self.results = self.results.append(result, ignore_index=True)
        self._save_results()

    def _save_results(self) -> None:
        if not self.path:
            return
        logging.debug(f'saving results file {self.path} ({len(self.results)} results)')
        self.results.to_pickle(f'{self.path}.pickle')
        self.results.to_csv(f'{self.path}.csv')

    def known_results(self) -> List[str]:
        return list(self.results.get('param_hash', []))

    def save_log(self, log: object, params: dict, parameter_hash: str, force: bool = False) -> None:
        if not self.log_path:
            return
        if force or datetime.now() - self.log_saved > timedelta(minutes=10):
            with open(self.log_path, 'wb') as f:
                pickle.dump((params, log), f)
                self.log_saved = datetime.now()


class FilesystemResultCollection(ResultCollection):

    EXTENSION = '.pickle'
    LOG_EXTENSION = '.log'

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path
        if not Path(path).exists():
            Path(path).mkdir()
        else:
            if not Path(path).is_dir():
                raise ValueError(f'Given storage path {Path(path).absolute()} must be '
                                 f'either non-existent or a directory.')

        self.log_saved = datetime.now()  # TODO Consider splitting throttling by parameter hash

    def add_result(self, parameter_hash: str, result: dict) -> None:
        with open(Path(self.path) / (parameter_hash + self.EXTENSION), 'wb') as f:
            pickle.dump(result, f)

    def _result_pickles(self) -> List[str]:
        return list(filter(lambda x: x.endswith(self.EXTENSION), os.listdir(self.path)))

    def known_results(self) -> List[str]:
        pickles = self._result_pickles()
        return [p.split('.')[0] for p in pickles]

    def save_log(self, log: object, params: dict, parameter_hash: str, force: bool = False) -> None:
        if force or datetime.now() - self.log_saved > timedelta(minutes=10):
            with open(Path(self.path) / (parameter_hash + self.EXTENSION + self.LOG_EXTENSION), 'wb') as f:
                pickle.dump((params, log), f)
                self.log_saved = datetime.now()

    def load_all(self) -> List[dict]:
        pickles = self._result_pickles()
        results = []
        for p in pickles:
            with open(Path(self.path) / p, 'rb') as f:
                results.append(pickle.load(f))
        return results


class MemoryResultCollection(ResultCollection):

    def __init__(self) -> None:
        super().__init__()
        self.results = {}
        self.log = {}

    def add_result(self, parameter_hash: str, result: dict) -> None:
        self.results[parameter_hash] = result

    def known_results(self) -> List[str]:
        return list(map(str, self.results.keys()))

    def save_log(self, log: object, params: dict, parameter_hash: str, force: bool = False) -> None:
        self.log[parameter_hash] = (params, log)

    def load_all(self) -> List[dict]:
        return list(self.results.values())


class StudyBase:

    def __init__(self, results: Optional[ResultCollection] = None, logging_callback: Optional[callable] = None,
                 randomize_order: bool = True) -> None:
        self._timer = {}

        if isinstance(results, str):
            results = FilesystemResultCollection(results)
        elif results is None:
            results = MemoryResultCollection()
        self.results = results

        self.log = None
        self.logging_callback = logging_callback

        self._cached_parameter_matrix = self.parameter_matrix()
        self._randomized_order = randomize_order
        if randomize_order:
            random.Random(42).shuffle(self._cached_parameter_matrix)
        self._current_params = None

        self.continue_on_error = False

    def _start_timer(self, name: str = 'default') -> None:
        self._timer[name] = datetime.now()

    def _stop_timer(self, name: str = 'default') -> float:
        return datetime.now() - self._timer[name]

    def _hash_parameters(self, params: dict) -> str:
        return sha256((self.__class__.__name__ + ': ' + str(params)).encode()).hexdigest()

    @staticmethod
    def parameter_matrix() -> List[dict]:
        raise NotImplementedError

    def primary_results(self, results: dict) -> dict:
        raise NotImplementedError

    def _add_result(self, params: dict, result: dict, memory: list = None) -> None:
        row = {}
        row.update({
            'parameters': list(map(str, params.keys())),
            'results': list(map(str, result.keys())),
            'experiment': self.__class__.__name__,
            'cpu': cpu,
            'timestamp': datetime.now(),
            'param_hash': self._hash_parameters(params),
            'memory': memory,
            # TODO consider adding git state
        })
        for env_var in [
            'OMP_NUM_THREADS',
            'NUMEXPR_NUM_THREADS',
            'MKL_NUM_THREADS',
            'TF_NUM_INTRAOP_THREADS',
            'TF_NUM_INTEROP_THREADS',
        ]:
            row.update({env_var: os.environ.get(env_var, None)})
        row.update(params)
        row.update(result)
        self.results.add_result(row['param_hash'], row)
        try:
            logging.debug(f'Added result: {self.primary_results(row)}')
        except NotImplementedError:
            logging.debug('Added result.')

    def _save_log(self, force: bool = False) -> None:
        if callable(self.logging_callback):
            self.logging_callback()
        self.results.save_log(self.log, self._current_params, self._hash_parameters(self._current_params), force)

    def run(self, **kwargs: dict) -> dict:
        raise NotImplementedError

    def run_single(self, params: dict) -> None:
        logging.debug(f'Running {self.__class__.__name__} for {params}')

        memory, result = memory_usage((self.run, [], params), retval=True)
        self._add_result(params, result, memory)

    def run_block(self, index: int, total: int) -> None:
        parameter_matrix = self._cached_parameter_matrix
        n = len(parameter_matrix)
        self.run_batch(parameter_matrix[int(index / total * n):int((index + 1) / total * n)])

    def run_all(self) -> None:
        self.run_batch(self._cached_parameter_matrix)

    def run_batch(self, batch: list) -> None:
        unfinished_parameters = [
            params for params in batch if self._hash_parameters(params) not in self.results.known_results()
        ]

        total = len(self._cached_parameter_matrix)
        unfinished = len(unfinished_parameters)
        finished = len(batch) - unfinished

        logging.debug(f'{self.__class__.__name__}: running {unfinished} unfinished jobs from a batch of {len(batch)} '
                      f'({finished} of this batch already completed, batch total {len(batch)}, '
                      f'study total {total} jobs)')

        ran = 0
        for params in unfinished_parameters:
            logging.debug(f'Progress: {ran/unfinished:.1%} session, {(ran + finished)/len(batch):.1%} batch, '
                          f'{(ran + finished)/total:.1%} total')
            try:
                ran += 1
                self._current_params = params
                self.run_single(params)
            except Exception as e:
                logging.debug(f'Running {self.__class__.__name__} resulted in {type(e)}: {e}.')
                if not self.continue_on_error:
                    raise
            finally:
                self._current_params = None

    @classmethod
    def cli(cls, args: list) -> None:
        logging.basicConfig(format='%(asctime)s.%(msecs)06d: %(levelname)s %(message)s',
                            level=logging.DEBUG,
                            datefmt='%Y-%m-%d %H:%M:%S')

        randomize = len(args) > 4
        study = cls(args[1], randomize_order=randomize)
        block_idx = int(args[2])
        block_total = int(args[3])
        study.run_block(block_idx, block_total)
