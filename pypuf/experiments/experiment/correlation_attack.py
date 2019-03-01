"""
This module provides an experiment class which learns an instance of LTFArray simulation PUF that uses
the lightweight-secure transform with the correlation attack learner.
"""
from numpy.random import RandomState
from numpy.linalg import norm
from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.regression.correlation_attack import CorrelationAttack
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.tools import TrainingSet, approx_dist
from math import ceil
from itertools import permutations
from scipy.stats import pearsonr


class ExperimentCorrelationAttack(Experiment):
    """
    This Experiment uses the CorrelationAttack learner on an LTFArray PUF simulation.
    """

    def __init__(self, n, k, progress_log_prefix, seed_model, seed_instance,
                 seed_challenge, seed_challenge_distance, N):
        """
        :param n: int
                  Number of stages of the PUF
        :param k: int
                  Number different LTFArrays
        :param progress_log_prefix: string
                  Prefix of the path or name of the experiment progress log file.
        :param seed_model: int
                  The seed which is used to initialize the pseudo-random number generator
                  which is used to generate the stage weights for the learner arbiter PUF simulation.
        :param seed_instance: int
                  The seed which is used to initialize the pseudo-random number generator
                  which is used to generate the stage weights for the arbiter PUF simulation.
        :param seed_challenge: int
                  The seed which is used to initialize the pseudo-random number generator
                  which is used to draft challenges for the TrainingSet.
        :param seed_challenge_distance: int
                  The seed which is used to initialize the pseudo-random number generator
                  which is used to draft challenges for the accuracy calculation.
        :param N: int
                  Number of challenges which are generated in order to learn the PUF simulation.
        """
        super().__init__(
            progress_log_name='%s.0x%x_0x%x_0_%i_%i_%i_%s_%s' % (
                progress_log_prefix,
                seed_model,
                seed_instance,
                n,
                k,
                N,
                LTFArray.transform_lightweight_secure.__name__,
                LTFArray.combiner_xor.__name__,
            ),
        )
        self.n = n
        self.k = k
        self.N = N
        self.seed_instance = seed_instance
        self.instance_prng = RandomState(seed=self.seed_instance)
        self.seed_model = seed_model
        self.model_prng = RandomState(seed=self.seed_model)
        self.combiner = LTFArray.combiner_xor
        self.transformation = LTFArray.transform_lightweight_secure
        self.seed_challenge = seed_challenge
        self.challenge_prng = RandomState(self.seed_challenge)
        self.seed_chl_distance = seed_challenge_distance
        self.distance_prng = RandomState(self.seed_chl_distance)
        self.instance = None
        self.learner = None
        self.model = None
        self.training_set = None
        self.validation_set = None

    def run(self):
        # TODO input transformation is computed twice. Add a shortcut to recycle results from the first computation
        self.instance = LTFArray(
            weight_array=LTFArray.normal_weights(self.n, self.k, random_instance=self.instance_prng),
            transform=self.transformation,
            combiner=self.combiner,
            bias=0.0
        )
        self.training_set = TrainingSet(instance=self.instance, N=int(ceil(self.N / 1.1)),
                                        random_instance=self.challenge_prng)
        self.validation_set = TrainingSet(instance=self.instance, N=int((self.N / 1.1) // 10),
                                          random_instance=self.distance_prng)
        self.learner = CorrelationAttack(
            n=self.n,
            k=self.k,
            training_set=self.training_set,
            validation_set=self.validation_set,
            weights_prng=self.model_prng,
            logger=self.progress_logger,
        )
        self.model = self.learner.learn()

    def analyze(self):
        """
        Analyzes the learned result.
        """
        assert self.model is not None

        def model_csv(model):
            return ','.join(map(str, model.weight_array.flatten() / norm(model.weight_array.flatten())))

        self.result_logger.info(
            # seed_instance  seed_model n      k      N      time   initial_iterations initial_accuracy best_accuracy
            '0x%x\t'        '0x%x\t'   '%i\t' '%i\t' '%i\t' '%f\t' '%i\t'             '%f\t'           '%f\t'
            # accuracy correct_iteration  best_iteration  rounds  permutation_accuracy   permutations
            # instance weights (norm.)  initial weights permuted weights final weights
            '%f\t'    '%s\t'               '%i\t'         '%s\t'       '%s\t'             '%s\t'        '%s\t'
            '%s\t'                '%s\t'           '%s',
            self.seed_instance,
            self.seed_model,
            self.n,
            self.k,
            self.validation_set.N + self.training_set.N,
            self.measured_time,
            self.learner.initial_iterations,
            self.learner.initial_accuracy,
            self.learner.best_accuracy,
            1.0 - approx_dist(
                self.instance,
                self.model,
                min(10000, 2 ** self.n),
                random_instance=self.distance_prng,
            ),
            str(self.find_correct_permutation(self.learner.initial_model.weight_array)) if
            self.learner.initial_accuracy > self.learner.OPTIMIZATION_ACCURACY_LOWER_BOUND else '',
            self.learner.best_iteration,
            self.learner.rounds,
            str(self.learner.approx_accuracy(self.learner.permuted_model)) if self.learner.permuted_model else '',
            ','.join(map(str, self.learner.permutations)) if self.learner.permutations else '',
            model_csv(self.instance),
            model_csv(self.learner.initial_model),
            model_csv(self.learner.permuted_model) if self.learner.permuted_model else '',
            model_csv(self.model)
        )

    def find_correct_permutation(self, weights):
        """
        Finds the best permutation of the given weights to fit the original instance weights.
        :param weights: The weight-array to permute
        :return: The best permutation
        """
        instance_weights = self.instance.weight_array

        max_correlation = 0
        best_permutation = None
        for permutation in list(permutations(range(self.k))):
            adopted_model_weights = self.learner.adopt_weights(weights, permutation)
            assert adopted_model_weights.shape == (self.k, self.n + 1), \
                'adopted weights shape is %s but was expected to be (%i, %i)' % (
                    str(adopted_model_weights.shape),
                    self.k,
                    self.n + 1
                )
            assert instance_weights.shape == (self.k, self.n + 1)
            correlation = [
                abs(pearsonr(
                    abs(adopted_model_weights[l] / norm(adopted_model_weights[l])),
                    abs(instance_weights[l] / norm(instance_weights[l]))
                )[0])
                for l in range(self.k)
            ]

            if sum(correlation) > max_correlation:
                max_correlation = sum(correlation)
                best_permutation = permutation

        return best_permutation
