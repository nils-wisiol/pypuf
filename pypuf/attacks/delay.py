import logging
from typing import Callable, Tuple, List

import numpy as np
import tensorflow as tf
from cma import CMA
from scipy.stats import pearsonr

from .. import random
from ..io import transform_challenge_11_to_01, ChallengeResponseSet
from ..metrics.common import approx_accuracy
from ..simulation.base import Simulation
from ..simulation.delay import LTFArray, XORArbiterPUF


class GapAttack:

    def __init__(self, crp_set: ChallengeResponseSet, k: int, transform: Callable,
                 pop_size: int = 20,
                 abort_delta: float = 5e-3,
                 abort_iter: int = 500,
                 fitness_threshold: float = .9,
                 pool_threshold: float = .9) -> None:
        self.crp_set = crp_set
        self.k = k
        self.n = crp_set.challenge_length
        self.transform = transform
        self.pop_size = pop_size
        self.abort_delta = abort_delta
        self.abort_iter = abort_iter
        self.fitness_threshold = fitness_threshold
        self.pool_threshold = pool_threshold
        self.pool = []
        self.attempt = 0
        self.traces = []

        self._objective_challenges = None

        if crp_set.responses.shape[1] != 1:
            raise ValueError(f'GapAttack only supports 1-bit responses, but responses of shape '
                             f'{crp_set.responses.shape} were given (expected (N, 1, r)).')

        # Compute PUF Reliabilities. These remain static throughout the optimization.
        self.puf_reliabilities = self.reliability_crp_set(self.crp_set.responses)
        self.puf_reliabilities_centered = self.puf_reliabilities - tf.reduce_mean(self.puf_reliabilities)

        # Linearize challenges for faster LTF computation (shape=(N,k,n))
        self._linearized_challenges = self.transform(self.crp_set.challenges, k=self.k)

    def _reliability_pearson(self, x: tf.Tensor) -> tf.Tensor:
        centered_x = x - tf.reduce_mean(x, axis=0)
        cov_xy = tf.tensordot(self.puf_reliabilities_centered, centered_x, axes=1)
        auto_cov = tf.sqrt(tf.reduce_sum(centered_x ** 2, axis=0) * tf.reduce_sum(self.puf_reliabilities_centered ** 2))
        corr = cov_xy / auto_cov
        return corr

    def _init_learner_state(self, seed: int) -> np.ndarray:
        state = np.empty(shape=(self.n + 1))
        state[:-1] = random.prng(f'GapAttack initial state for seed {seed}').normal(0, 1, size=self.n)  # init weights
        state[-1] = 2  # init eps
        return state

    @staticmethod
    def reliability_model(delay_diffs: tf.Tensor, eps: tf.Tensor) -> tf.Tensor:
        return tf.math.greater(tf.transpose(tf.abs(delay_diffs)), eps)

    @staticmethod
    def reliability_crp_set(responses: np.ndarray) -> tf.Tensor:
        # Convert to 0/1 from 1/-1
        N, _, r = responses.shape
        responses = responses.reshape(N, r)
        responses = transform_challenge_11_to_01(responses)
        return tf.constant(np.abs(responses.shape[1] / 2 - np.sum(responses, axis=-1)), dtype=tf.double)

    def objective(self, state: tf.Tensor, challenges_T: tf.Tensor, fitness_history: List[tf.Tensor]) -> tf.Tensor:
        # Weights and epsilon have the first dim as number of population
        weights = state[:, :self.n]
        eps = state[:, -1]

        # compute model reliabilities
        delay_diffs = tf.linalg.matmul(weights, challenges_T)
        model_reliabilities = self.reliability_model(delay_diffs, eps)

        # compute pearson correlation coefficients with target reliabilities
        x = tf.cast(model_reliabilities, tf.double)
        fitness = tf.abs(self._reliability_pearson(x) - tf.constant(1, dtype=tf.float64))

        # record fitness
        fitness_history.append(tf.reduce_min(fitness))

        return fitness

    def learn(self, seed: int, max_retries: int = np.inf) -> Simulation:
        current_attempt = 0

        # main learning loop
        while len(self.pool) < self.k and current_attempt < max_retries:
            chain_seed = random.seed(f'GapAttack on seed {seed} attempt {self.attempt}')
            logging.debug(f'Attempting to learn {len(self.pool) + 1}-th chain with seed {chain_seed} '
                          f'(attempt {self.attempt})')

            weights, eps, fitness, fitness_hist, abort_reasons = self.learn_chain(len(self.pool), chain_seed)
            discard_reasons = self.discard_chain(weights, fitness, self.pool)
            logging.debug(f'Found chain with fitness {fitness:.3f}, aborting due to {", ".join(abort_reasons)}')

            self.traces.append({
                'weights': weights,
                'eps': eps,
                'fitness': fitness,
                'fitness_hist': fitness_hist,
                'abort_reasons': abort_reasons,
                'discard_reasons': discard_reasons,
            })

            self.attempt += 1
            current_attempt += 1

            if discard_reasons:
                logging.debug('Discarding chain due to ' + ', '.join(discard_reasons))
            else:
                self.pool.append(weights)
                logging.debug(f'Adding {len(self.pool)}-th chain to pool, {self.k - len(self.pool)} still missing')

        if self.pool:
            # if training accuracy < 0.5, we flip a chain to flip all outputs
            model = LTFArray(np.array(self.pool), self.transform)
            if np.average(approx_accuracy(model, self.crp_set)) < .5:
                model.weight_array[0] *= -1
        else:
            # if we couldn't find a single chain, just provide a random model to not break the interface
            model = XORArbiterPUF(
                n=self.n, k=self.k,
                seed=random.seed(f'GapAttack on seed {seed} no result guess'),
                transform=self.transform
            )

        return model

    def learn_chain(self, l: int, seed: int, callback: Callable = None,
                    cma_kwargs: dict = None) -> Tuple[np.ndarray, float, float, List[float], List[str]]:

        # prepare specialized objective function
        objective_challenges_T = np.array(self._linearized_challenges[:, l, :], dtype=np.float64).T
        fitness_history = []

        def objective(state: tf.Tensor) -> tf.Tensor:
            return self.objective(state, objective_challenges_T, fitness_history)

        # seed tensorflow for reproducible results
        tf.random.set_seed(random.seed(f'GapAttack tensorflow seed for seed {seed}'))

        # initial learning state
        init_state = self._init_learner_state(seed)

        # prepare callback function, will be called after each generation
        def callback_hook(_cma: CMA, logger: logging.Logger) -> None:
            if _cma.generation % (self.abort_iter // 10) == 0:
                logger.info(f'generation {_cma.generation:n}/{self.abort_iter:n} '
                            f'({_cma.generation/self.abort_iter:.1%})')
            if callback:
                callback(_cma, _cma.m[:-1], _cma.m[-1])

        # initialize learner
        cma = CMA(
            initial_solution=init_state,
            initial_step_size=1.0,
            fitness_function=objective,
            termination_no_effect=self.abort_delta,
            callback_function=callback_hook,
            **(cma_kwargs or {}),
        )

        # with tf.device('/GPU:0'):
        # learn
        w, fitness = cma.search(max_generations=self.abort_iter)

        # extract results
        weights, eps = w[:-1], w[-1]
        termination_reasons = [key for key, val in cma.should_terminate(True)[1].items() if val]

        return weights, eps, fitness, fitness_history, termination_reasons

    def discard_chain(self, weights: np.ndarray, fitness: float, pool: List[np.ndarray]) -> List[str]:
        discard_reasons = []

        # normalize weights to have positive first weight
        if weights[0] < 0:
            weights *= -1

        # check if fitness is sufficient (below)
        if fitness >= self.fitness_threshold or np.isnan(fitness) or np.isinf(fitness):
            discard_reasons.append('fitness too high')

        # Check if learned model (w) is a 'new' chain (not correlated to other chains)
        for i, v in enumerate(pool):
            corr = tf.abs(pearsonr(weights, v)[0])
            if corr > self.pool_threshold:
                discard_reasons.append(f'too similar, to learned chain {i} (correlation {corr.numpy():.2f})')

        return discard_reasons
