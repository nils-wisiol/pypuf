import time
from numpy.random import RandomState
from numpy import amin, amax, mean, array, append
from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.regression.logistic_regression import LogisticRegression
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf import tools


class ExperimentLogisticRegression(Experiment):
    """
        This Experiment uses the logistic regression learner on an LTFArray PUF simulation.
    """

    def __init__(self, log_name, n, k, crp_count, seed_instance, seed_model, transformation, combiner,
                 restarts=float('inf'),
                 convergence=1.1):
        self.log_name = log_name
        self.n = n
        self.k = k
        self.N = crp_count
        self.seed_instance = seed_instance
        self.instance_prng = RandomState(seed=self.seed_instance)
        self.seed_model = seed_model
        self.model_prng = RandomState(seed=self.seed_model)
        self.restarts = restarts
        self.convergence = convergence
        self.combiner = combiner
        self.transformation = transformation
        self.instance = LTFArray(
            weight_array=LTFArray.normal_weights(n, k, random_instance=self.instance_prng),
            transform=self.transformation,
            combiner=self.combiner,
        )
        self.learner = LogisticRegression(
            tools.TrainingSet(instance=self.instance, N=crp_count),
            n,
            k,
            transformation=self.transformation,
            combiner=self.combiner,
            weights_prng=self.model_prng
        )
        super().__init__(self.log_name, self.learner)
        self.min_dist = 0
        self.test_dist = 0
        self.accuracy = array([])
        self.training_times = array([])
        self.iterations = array([])
        self.dist = 1.0

    def name(self):
        return 'ExperimentLogisticRegression n={0} k={1} N={2} ' \
               'seed_instance={3}, seed_model={4}'.format(self.n, self.k,
                                                          self.N,
                                                          self.seed_instance,
                                                          self.seed_model)

    def output_string(self):
        msg = 'training times: {0}\n' \
              'iterations: {1}\n' \
              'test accuracy: {2}\n' \
              'min/avf/max training time: {3} / {4} / {5}\n' \
              'min/avg/max iteration count: {6} / {7} / {8}\n' \
              'min/avg/max test accuracy: {9} / {10} / {11}'.format(
            self.training_times,
            self.iterations,
            self.accuracy,
            amin(self.training_times),
            mean(self.training_times),
            amax(self.training_times),
            amin(self.iterations),
            mean(self.iterations),
            amax(self.iterations),
            amin(self.accuracy),
            mean(self.accuracy),
            amax(self.accuracy),
        )
        return msg

    def analysis(self):
        """
            This method learns one instance self.restarts times or a self.convergence threshold is reached.
            The results are saved in:
                self.training_times
                self.dist
                self.accuracy
                self.iterations
        :return:
        """
        i = 0.0
        while i < self.restarts and 1.0 - self.dist < self.convergence:
            start = time.time()
            model = self.learner.learn()
            end = time.time()
            self.training_times = append(self.training_times, end - start)
            self.dist = tools.approx_dist(self.instance, model, min(10000, 2 ** self.n))
            self.accuracy = append(self.accuracy, 1.0 - self.dist)
            self.iterations = append(self.iterations, self.learner.iteration_count)
            i += 1
