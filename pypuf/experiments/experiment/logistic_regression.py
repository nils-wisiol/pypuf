from numpy.random import RandomState
from numpy import amin, amax, mean, array, append
from numpy.linalg import norm
from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.regression.logistic_regression import LogisticRegression
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf import tools


class ExperimentLogisticRegression(Experiment):
    """
        This Experiment uses the logistic regression learner on an LTFArray PUF simulation.
    """

    def __init__(self, log_name, n, k, N, seed_instance, seed_model, transformation, combiner):
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
            tools.TrainingSet(instance=self.instance, N=self.N),
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
            '0x%x\t'        '0x%x\t'   '%i\t' '%i\t' '%i\t' '%i\t' '%s\t' '%s\t' '%i\t' '%f\t' '%f\t'    '%s' % (
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
                1.0 - tools.approx_dist(self.instance, self.model, min(10000, 2 ** self.n)),
                ','.join(map(str, self.model.weight_array.flatten() / norm(self.model.weight_array.flatten())))
            )
        )
