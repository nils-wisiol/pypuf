from numpy import zeros, newaxis
from numpy.random import RandomState
from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.mlp import MLP
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf import tools


class ExperimentMLP(Experiment):
    """
    This Experiment uses the MLP learner on an LTFArray PUF simulation.
    """

    def __init__(
            self, log_name, n, k, N, seed_simulation, seed_model, transformation, combiner, seed_challenge=0x5A551,
            seed_accuracy=0xB055, batch_size=1000, iteration_limit=1000
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
        :param seed_simulation: int
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
        :param seed_accuracy: int default is 0xB055
                                  The seed which is used to initialize the pseudo-random number generator
                                  which is used to draft challenges for the accuracy calculation.
        """
        super().__init__(
            log_name='%s_MLP_0x%x_0x%x_0_%i_%i_%i_%s_%s' % (
                log_name,
                seed_model,
                seed_simulation,
                n,
                k,
                N,
                transformation,
                combiner,
            ),
        )
        self.n = n
        self.k = k
        self.N = N
        self.seed_simulation = seed_simulation
        self.seed_model = seed_model
        self.model_prng = RandomState(seed=self.seed_model)
        self.combiner = combiner
        self.transformation = transformation
        self.seed_challenges = seed_challenge
        self.challenge_prng = RandomState(self.seed_challenges)
        self.seed_accuracy = seed_accuracy
        self.batch_size = batch_size
        self.iteration_limit = iteration_limit
        self.simulation = None
        self.learner = None
        self.model = None

    def prepare(self):
        """
        Prepare learning: initialize learner, prepare training set, etc.
        """
        self.simulation = LTFArray(
            weight_array=LTFArray.normal_weights(
                self.n,
                self.k,
                300,
                40,
                random_instance=RandomState(seed=self.seed_simulation)
            ),
            transform=self.transformation,
            combiner=self.combiner,
        )
        self.learner = MLP(
            N=self.N,
            n=self.n,
            k=self.k,
            simulation=self.simulation,
            seed_challenges=self.seed_challenges,
            seed_model=self.seed_model,
            batch_size=self.batch_size,
            iteration_limit=self.iteration_limit,
        )
        self.learner.prepare()

    def run(self):
        """
        Runs the learner
        """
        self.model = self.learner.learn()

    def analyze(self):
        """
        Analyzes the learned result.
        """
        assert self.model is not None
        distance = 1.0 - tools.approx_dist(
                self.simulation,
                self.model,
                min(10000, 2 ** self.n),
                random_instance=RandomState(seed=self.seed_accuracy)
            )
        self.result_logger.info(
            '0x%x\t0x%x\t%i\t%i\t%i\t%f\t%s\t%s\t%s\t%i\t%s',
            self.seed_simulation,
            self.seed_model,
            self.n,
            self.k,
            self.N,
            max(distance, 1 - distance),
            self.measured_time,
            self.transformation,
            self.combiner,
            self.learner.clf.n_iter_,
            ','.join(map(str, self.learner.clf.loss_curve_)),
        )
