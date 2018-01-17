"""
This module provides an experiment class which learns an instance of LTFArray simulation PUF with the logistic
regression learner.
"""
from numpy.random import RandomState
from numpy.linalg import norm
from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.regression.logistic_regression import LogisticRegression
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf import tools


class ExperimentLogisticRegression(Experiment):
    """
    This Experiment uses the logistic regression learner on an LTFArray PUF simulation.
    """

    def __init__(
            self, log_name, n, k, N, seed_instance, seed_model, transformation, combiner, seed_challenge=0x5A551,
            seed_chl_distance=0xB055,
    ):
        """
        :param log_name: string
                         Prefix of the path or name of the experiment log file.
        :param n: int
                  Number of stages of the PUF
        :param k: int
                  Number different LTFArrays
        :param N: int
                  Number of challenges which are generated in order to learn the PUF simulation.
        :param seed_instance: int
                              The seed which is used to initialize the pseudo-random number generator
                              which is used to generate the stage weights for the arbiter PUF simulation.
        :param seed_model: int
                           The seed which is used to initialize the pseudo-random number generator
                           which is used to generate the stage weights for the learner arbiter PUF simulation.
        :param transformation: A function: array of int with shape(N,k,n), int number of PUFs k -> shape(N,k,n)
                               The function transforms input challenges in order to increase resistance against attacks.
        :param combiner: A function: array of int with shape(N,k,n) -> array of in with shape(N)
                         The functions combines the outputs of k PUFs to one bit results,
                         in oder to increase resistance against attacks.
        :param seed_challenge: int default is 0x5A551
                               The seed which is used to initialize the pseudo-random number generator
                               which is used to draft challenges for the TrainingSet.
        :param seed_chl_distance: int default is 0xB055
                                  The seed which is used to initialize the pseudo-random number generator
                                  which is used to draft challenges for the accuracy calculation.
        """
        super().__init__(
            log_name='%s.0x%x_0x%x_0_%i_%i_%i_%s_%s' % (
                log_name,
                seed_model,
                seed_instance,
                n,
                k,
                N,
                transformation.__name__,
                combiner.__name__,
            ),
        )
        self.n = n
        self.k = k
        self.N = N
        self.seed_instance = seed_instance
        self.instance_prng = RandomState(seed=self.seed_instance)
        self.seed_model = seed_model
        self.model_prng = RandomState(seed=self.seed_model)
        self.combiner = combiner
        self.transformation = transformation
        self.seed_challenge = seed_challenge
        self.challenge_prng = RandomState(self.seed_challenge)
        self.seed_chl_distance = seed_chl_distance
        self.distance_prng = RandomState(self.seed_chl_distance)
        self.instance = None
        self.learner = None
        self.model = None

    def run(self):
        """
        Initializes the instance, the training set and the learner to then run the logistic regression
        with the given parameters.
        """
        # TODO input transformation is computed twice. Add a shortcut to recycle results from the first computation
        self.instance = LTFArray(
            weight_array=LTFArray.normal_weights(self.n, self.k, random_instance=self.instance_prng),
            transform=self.transformation,
            combiner=self.combiner,
        )
        self.learner = LogisticRegression(
            tools.TrainingSet(instance=self.instance, N=self.N, random_instance=self.challenge_prng),
            self.n,
            self.k,
            transformation=self.transformation,
            combiner=self.combiner,
            weights_prng=self.model_prng,
            logger=self.progress_logger,
        )
        self.model = self.learner.learn()

    def analyze(self):
        """
        Analyzes the learned result.
        """
        assert self.model is not None

        self.result_logger.info(
            # seed_instance  seed_model i      n      k      N      trans  comb   iter   time   accuracy  model values
            '0x%x\t'        '0x%x\t'   '%i\t' '%i\t' '%i\t' '%i\t' '%s\t' '%s\t' '%i\t' '%f\t' '%f\t'    '%s',
            self.seed_instance,
            self.seed_model,
            0,  # restart count, kept for compatibility to old log files
            self.n,
            self.k,
            self.N,
            self.transformation.__name__,
            self.combiner.__name__,
            self.learner.iteration_count,
            self.measured_time,
            1.0 - tools.approx_dist(
                self.instance,
                self.model,
                min(10000, 2 ** self.n),
                random_instance=self.distance_prng,
            ),
            ','.join(
                ['%.12f' % x for x in self.model.weight_array.flatten() / norm(self.model.weight_array.flatten())]
            ),
        )
