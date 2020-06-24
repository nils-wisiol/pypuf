import logging
import os
from datetime import datetime
from hashlib import sha256
from typing import List

import pandas as pd
from numpy.distutils import cpuinfo

# noinspection PyBroadException
try:
    cpu = cpuinfo.cpu.info[0]['model name']
except Exception:
    cpu = None


class StudyBase:

    def __init__(self, results_file: str) -> None:
        self._timer = {}

        self.results_file = results_file
        self.results = None
        self._load_results()

        self._cached_parameter_matrix = self.parameter_matrix()

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

    def _add_result(self, params: dict, result: dict) -> None:
        row = {}
        row.update({
            'experiment': self.__class__.__name__,
            'cpu': cpu,
            'timestamp': datetime.now(),
            'param_hash': self._hash_parameters(params),
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
        self.results = self.results.append(row, ignore_index=True)
        try:
            logging.debug(f'Adding result: {self.primary_results(row)}')
        except NotImplementedError:
            pass
        self._save_results()

    def _load_results(self) -> None:
        logging.debug(f'loading results file {self.results_file}')
        try:
            self.results = pd.read_pickle(self.results_file)
        except FileNotFoundError:
            self.results = pd.DataFrame()

    def _save_results(self) -> None:
        logging.debug(f'saving results file {self.results_file} ({len(self.results)} results)')
        self.results.to_pickle(self.results_file)
        self.results.to_csv(f'{self.results_file}.csv')

    def _known_parameter_hashes(self) -> List[str]:
        return list(self.results.get('param_hash', []))

    def run(self, **kwargs: dict) -> dict:
        raise NotImplementedError

    def run_single(self, params: dict) -> None:
        logging.debug(f'Running {self.__class__.__name__} for {params}')
        self._add_result(params, self.run(**params))

    def run_block(self, index: int, total: int) -> None:
        n = len(self._cached_parameter_matrix)
        self.run_batch(self._cached_parameter_matrix[int(index / total * n):int((index + 1) / total * n)])

    def run_all(self) -> None:
        self.run_batch(self._cached_parameter_matrix)

    def run_batch(self, batch: list) -> None:
        unfinished_parameters = [
            params for params in batch if self._hash_parameters(params) not in self._known_parameter_hashes()
        ]

        total = len(self._cached_parameter_matrix)
        unfinished = len(unfinished_parameters)
        finished = len(batch) - unfinished

        logging.debug(f'{self.__class__.__name__}: running {unfinished} unfinished jobs from a batch of {len(batch)} '
                      f'({finished} of this batch already completed, batch total {len(batch)}, '
                      f'study total {total} jobs)')

        ran = 0
        for params in unfinished_parameters:
            logging.debug(f'Progress: {ran/unfinished:.1%} session, {(ran + finished)/total:.1%} batch, '
                          f'{(ran + finished)/total:.1%} total')
            try:
                ran += 1
                self.run_single(params)
            except Exception as e:
                logging.debug(f'Running {self.__class__.__name__} resulted in {e}.')
                if not self.continue_on_error:
                    raise

    @classmethod
    def cli(cls, args: list) -> None:
        logging.basicConfig(format='%(asctime)s.%(msecs)06d: %(levelname)s %(message)s',
                            level=logging.DEBUG,
                            datefmt='%Y-%m-%d %H:%M:%S')

        study = cls(args[1])
        block_idx = int(args[2])
        block_total = int(args[3])
        study.run_block(block_idx, block_total)
