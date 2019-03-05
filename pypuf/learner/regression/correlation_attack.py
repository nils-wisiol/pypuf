"""
This module provides an attack on XOR Arbiter PUFs that is based off known correlation in sub-challenge generation of
the input transformation.
"""
from copy import deepcopy
from itertools import permutations
from scipy.io import loadmat
from numpy.random import RandomState
from numpy import empty, roll, count_nonzero, sign, zeros
from pypuf.learner.base import Learner
from pypuf.learner.regression.logistic_regression import LogisticRegression
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.tools import ChallengeResponseSet
from collections import namedtuple

PermData = namedtuple('Permutation', ['permutation', 'accuracy'])


class CorrelationAttack(Learner):
    """
    Learn an LTF-Array that uses the transform_lightweight_secure.

    This Attack uses Logistic Regression (LR) to learn an initial model and subsequently exploits the correlation
    between the weight arrays to restart the LR learner on a permuted model. This leads to a
    faster retrieval of high-accuracy models than pure LR.
    """
    OPTIMIZATION_ACCURACY_LOWER_BOUND = .65
    OPTIMIZATION_ACCURACY_UPPER_BOUND = .95
    OPTIMIZATION_ACCURACY_GOAL = .98

    def __init__(self, n, k, training_set, validation_set, weights_mu=0, weights_sigma=1, weights_prng=RandomState(),
                 lr_iteration_limit=1000, mini_batch_size=0, convergence_decimals=2, shuffle=False, logger=None):
        """
        Initialize a Correlation Attack Learner for the specified LTF Array which uses transform_lightweight_secure.

        :param n: Input length
        :param k: Number of parallel LTFs in the LTF Array
        :param training_set: The training set, i.e. a data structure containing challenge response pairs
        :param validation_set: The validation set, i.e. a data structure containing challenge response pairs. Used for
        approximating accuracies of permuted models (can be smaller e.g. 0.1*training_set_size)
        :param weights_mu: mean of the Gaussian that is used to choose the initial model
        :param weights_sigma: standard deviation of the Gaussian that is used to choose the initial model
        :param weights_prng: PRNG to draw the initial model from. Defaults to fresh `numpy.random.RandomState` instance.
        :param lr_iteration_limit: Iteration limit for a single LR learner run
        :param logger: logging.Logger
                       Logger which is used to log detailed information of learn iterations.
        """
        self.n = n
        self.k = k

        self.validation_set_efba = ChallengeResponseSet(
            challenges=LTFArray.efba_bit(LTFArray.transform_lightweight_secure(validation_set.challenges, k)),
            responses=validation_set.responses
        )

        self.logger = logger

        self.lr_learner = LogisticRegression(
            t_set=training_set,
            n=n,
            k=k,
            transformation=LTFArray.transform_lightweight_secure,
            combiner=LTFArray.combiner_xor,
            weights_mu=weights_mu,
            weights_sigma=weights_sigma,
            weights_prng=weights_prng,
            logger=logger,
            iteration_limit=lr_iteration_limit,
            minibatch_size=mini_batch_size,
            convergence_decimals=convergence_decimals,
            shuffle=shuffle
        )

        self.initial_accuracy = .5
        self.initial_lr_iterations = 0
        self.initial_model = None
        self.total_lr_iterations = 0
        self.best_permutation_iteration = 0
        self.total_permutation_iterations = 0
        self.best_permutation = None
        self.best_accuracy = None

        assert n in (64, 128), 'Correlation attack for %i bit is currently not supported.' % n
        assert validation_set.N >= 1000, 'Validation set should contain at least 1000 challenges.'

        self.correlation_permutations = loadmat(
            'data/correlation_permutations_lightweight_secure_original_%i_10.mat' % n
        )['shiftOverviewData'][:, :, 0].astype('int64')

    def learn(self):
        """
        Compute a model according to the given LTF Array parameters and training set.
        Note that this function can take long to return.
        :return: pypuf.simulation.arbiter_based.LTFArray
                 The computed model.
        """
        self.initial_model = initial_model = self.lr_learner.learn()
        self.logger.debug('initial weights for corr attack:')
        self.logger.debug(','.join(map(str, initial_model.weight_array.flatten())))
        self.initial_accuracy = self.approx_accuracy(initial_model, self.validation_set_efba.block_subset(0, 2))
        self.initial_lr_iterations = self.lr_learner.iteration_count
        self.total_lr_iterations = self.initial_lr_iterations
        initial_updater = self.lr_learner.updater

        self.best_accuracy = self.initial_accuracy

        self.logger.debug('Initial accuracy is %.4f' % self.initial_accuracy)

        if self.initial_accuracy < self.OPTIMIZATION_ACCURACY_LOWER_BOUND:
            self.logger.debug('initial learning below threshold, aborting')
            return initial_model

        if self.initial_accuracy > self.OPTIMIZATION_ACCURACY_GOAL:
            self.logger.debug('initial learning successful, aborting')
            return initial_model

        # Try all permutations with high initial accuracy and see if any of them lead to a good final result
        high_accuracy_permutations = self.find_high_accuracy_weight_permutations(
            initial_model.weight_array,
            # allow some accuracy loss by permuting
            # the higher the initial accuracy, the higher the loss we allow
            # result will never be below 0.925
            1.2 * self.best_accuracy - .2
        )

        best_model = initial_model
        self.logger.debug('Trying %i permuted weights.' % len(high_accuracy_permutations))
        for (iteration, perm_data) in enumerate(high_accuracy_permutations):
            self.total_permutation_iterations += 1
            weights = self.adopt_weights(initial_model.weight_array, perm_data.permutation)
            self.lr_learner.updater = deepcopy(initial_updater)
            self.lr_learner.updater.step_size *= 10
            model = self.lr_learner.learn(init_weight_array=weights, refresh_updater=False)
            self.total_lr_iterations += self.lr_learner.iteration_count
            accuracy = self.approx_accuracy(model, self.validation_set_efba.block_subset(1, 2))
            self.logger.debug(
                'With permutation no %d=%s, after restarting the learning we achieved accuracy %.4f -> %.4f!' %
                (iteration, perm_data.permutation, perm_data.accuracy, accuracy))
            if accuracy > 0.1 + 0.9 * self.initial_accuracy \
                    and accuracy > self.best_accuracy:
                # demand some "substantial" improvement of accuracy
                # what substantial means becomes weaker as we approach
                # perfect accuracy
                best_model = model
                self.best_accuracy = accuracy
                self.best_permutation_iteration = iteration + 1
                self.best_permutation = perm_data.permutation
            else:
                self.logger.debug('Learning after permuting lead to accuracy %.2f, no improvement :-(' % accuracy)

            if accuracy > self.OPTIMIZATION_ACCURACY_GOAL:
                self.logger.debug('Found a model with accuracy better than %.2f. Terminating' %
                                  self.OPTIMIZATION_ACCURACY_GOAL)
                return model

        self.logger.debug('After trying all permutations, we found a model with acc. %.2f.' % self.best_accuracy)
        return best_model

    def find_high_accuracy_weight_permutations(self, weights, threshold):
        """
        Gives permutations for the weight-array resulting in the highest model accuracies.
        :param weights: The original weight-array
        :param threshold: Minimum accuracy to consider
        :return: The 5k permutations with the highest accuracy
        """
        high_accuracy_permutations = []
        adopted_instance = LTFArray(
            weight_array=zeros((self.k, self.n)),
            transform=LTFArray.transform_lightweight_secure,
            combiner=LTFArray.combiner_xor
        )
        for permutation in list(permutations(range(self.k)))[1:]:
            adopted_instance.weight_array = self.adopt_weights(weights, permutation)
            accuracy = self.approx_accuracy(adopted_instance)
            self.logger.debug('For permutation %s, we have accuracy %.4f' % (permutation, accuracy))
            if accuracy >= threshold:
                high_accuracy_permutations.append(PermData(permutation, accuracy))

        high_accuracy_permutations.sort(key=lambda x: -x.accuracy)
        return high_accuracy_permutations[:5 * self.k]

    def approx_accuracy(self, instance, efba_set=None):
        """
        Approximate the accuracy of the instance on the given set.
        :param instance: pypuf.simulation.arbiter_based.LTFArray
        :param efba_set: A challenge-response-set containing efba sub-challenges (default: self.validation_set_efba)
        :return: Accuracy of the instance
        """
        if efba_set is None:
            efba_set = self.validation_set_efba
        size = efba_set.N
        responses = sign(instance.combiner(instance.core_eval(efba_set.challenges)))
        return count_nonzero(responses == efba_set.responses) / size

    def adopt_weights(self, weights, permutation):
        """
        Adopts the weights with the given permutation exploiting the correlations of the lightweight-secure transform.
        :param weights: A weight-array of an LTFArray
        :param permutation: Permutation as returned from itertools.permutations
        :return: Permuted weight-array
        """
        adopted_weights = empty(weights.shape)
        for l in range(self.k):
            adopted_weights[permutation[l], :] = \
                roll(weights[l, :], self.correlation_permutations[l, permutation[l]])
        return adopted_weights
