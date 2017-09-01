from numpy import array, zeros
from numpy.random import RandomState
from numpy.linalg import norm
from pypuf.experiments.experiment.base import Experiment
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, NoisyLTFArray, SimulationMajorityLTFArray
from pypuf.learner.evolution_strategies.becker import Reliability_based_CMA_ES
from pypuf import tools


class ExperimentReliabilityBasedCMAES(Experiment):
    """
        This class is designed to use reliability based CMA-ES learning on NoisyLFTArray and SimulationMajorityLTFArray
        simulations.
    """

    def __init__(self, log_name, n, k, challenge_count, seed_instance, seed_instance_noise, seed_model, transformation,
                 combiner, mu, sigma, sigma_noise, vote_count, repetitions, limit_step_size, limit_iteration,
                 seed_challenges, bias=False):
        """
        :param log_name: string
                         Prefix of the experiment logfile.
        :param n: int
                  The number of stages each arbiter PUF have.
        :param k: int
                  The number of arbiter PUFs
        :param challenge_count: int
                  The number which indicates how many challenges are used to create an instance.
        :param seed_instance: int
                              Random seed which is used to initialize the pseudo-random number generator
                              which is used to generate the stage weights for the arbiter PUF simulation.
        :param seed_instance_noise: int
                                    Random seed which is used to initialize the pseudo-random number generator
                                    which is used to generate the noise for the arbiter PUF simulation.
        :param seed_model: int
                           Random seed which is used to initialize the pseudo-random number generator
                           which is used for the reliability based CMA-ES.
        :param transformation: A function: array of int with shape(N,k,n), int number of PUFs k -> shape(N,k,n)
                               The function transforms input challenges in order to increase resistance against attacks.
        :param combiner: A function: array of int with shape(N,k,n) -> array of in with shape(N)
                         The functions combines the outputs of k PUFs to one bit results,
                         in oder to increase resistance against attacks.
        :param mu: float
                   Mean (“centre”) of the stage weight distribution of the PUF instance simulation.
        :param sigma: float
                      Standard deviation of the stage weight distribution of the PUF instance simulation.
        :param sigma_noise: float
                            Standard deviation of noise distribution of the PUF instance simulation.
        :param vote_count: int
                           A positive odd number of votes of the PUF instance simulation. It also decides which kind of
                           PUF instance simulation is used. If vote_count == 1 a NoisyLFTArray instance is used.
                           If votes_count > 1 and odd a SimulationMajorityLTFArray instance is used.
        :param repetitions: int
                            Number how many times a challenge is evaluated by the PUF instance simulation.
        :param limit_step_size: float
                                A positive number which is the maximal step size of CMA-ES.
        :param limit_iteration: int
                                The Maximum number of iterations within the CMA-ES evolutionary_search method.
        :param seed_challenges: int
                           Random seed which is used to initialize the pseudo-random number generator
                           which is used for the generation of challenges.
        :param bias: boolean
                     This value is used to turn on input/output distort of the  PUF instance simulation.
        """
        super().__init__(log_name='%s.0x%x_0x%x_0_%x_0_%x_0_%i_%i_%i_%i_%i_%s_%s' % (
            log_name,
            seed_model,
            seed_instance,
            seed_instance_noise,
            seed_challenges,
            n,
            k,
            challenge_count,
            vote_count,
            repetitions,
            transformation.__name__,
            combiner.__name__,

        ),
                         )
        self.log_name = log_name
        self.n = n
        self.k = k
        self.N = challenge_count
        self.mu = mu
        self.sigma = sigma
        self.sigma_noise = sigma_noise
        self.repetitions = repetitions
        self.limit_step_size = limit_step_size
        self.limit_iteration = limit_iteration
        self.bias = bias
        self.seed_instance = seed_instance
        self.seed_instance_noise = seed_instance_noise
        self.seed_model = seed_model
        self.seed_challenges = seed_challenges
        self.transformation = transformation
        self.combiner = combiner
        self.vote_count = vote_count
        self.instance = None
        self.learner = None
        self.challenges = None
        self.responses_repeated = None
        self.model = None

    def run(self):
        """
        This method setups and executes the puf simulation and the learner. The method decides whether NoisyLFTArray
        in case of self.vote_count == 1 or SimulationMajorityLTFArray in case self.vote_count > 1 should be used.
        """
        # Random number generators
        instance_prng = RandomState(self.seed_instance)
        noise_prng = RandomState(self.seed_instance_noise)
        model_prng = RandomState(self.seed_model)
        challenge_prng = RandomState(self.seed_challenges)

        # Weight array for the instance which should be learned
        weight_array = LTFArray.normal_weights(self.n, self.k, self.mu, self.sigma, random_instance=instance_prng)

        # vote_count must be an odd number greater then zero
        assert self.vote_count > 0 and self.vote_count % 2 != 0
        # Decide which instance should be used, a NoisyLFTArray or a SimulationMajorityLTFArray.
        if self.vote_count == 1:
            self.instance = NoisyLTFArray(weight_array, self.transformation, self.combiner, self.sigma_noise,
                                          random_instance=noise_prng, bias=self.bias)
        else:
            self.instance = SimulationMajorityLTFArray(weight_array, self.transformation, self.combiner,
                                                       self.sigma_noise, random_instance_noise=noise_prng,
                                                       bias=self.bias, vote_count=self.vote_count)

        # sample challenges
        self.challenges = array(list(tools.sample_inputs(self.n, self.N, random_instance=challenge_prng)))

        # extract responses from instance
        self.responses_repeated = zeros((self.repetitions, self.N))
        for i in range(self.repetitions):
            self.responses_repeated[i, :] = self.instance.eval(self.challenges)

        # Setup learner
        self.learner = Reliability_based_CMA_ES(self.k, self.n, self.transformation, self.combiner, self.challenges,
                                                self.responses_repeated, self.repetitions,
                                                self.limit_step_size,
                                                self.limit_iteration, prng=model_prng)
        self.model = self.learner.learn()

    def analyze(self):
        """
        This method is used to analyse the learned model of the self.run() method.
        It summarizes the results and logs them into the experiment and experimenter log.
        """
        assert self.model is not None

        responses_model = self.model.eval(self.challenges)
        responses_instance = Reliability_based_CMA_ES.get_common_responses(self.responses_repeated)
        assert len(responses_model) == len(responses_instance)
        accuracy = 1.0 - tools.approx_dist(self.instance, self.model, min(10000, 2 ** self.n),
                                           random_instance=RandomState(0xC0DEBA5E))
        abortions = self.learner.abortions

        # seed_instance  seed_model       i      n      k      N  vote_count  trans  comb   iter   time   accuracy abortions model values
        msg = '0x%x\t'        '0x%x\t'   '%i\t' '%i\t' '%i\t' '%i\t' '%i\t' '%s\t' '%s\t' '%i\t' '%f\t' '%f\t' '%f\t'    '%s' % (
            self.seed_instance,
            self.seed_model,
            0,  # restart count, kept for compatibility to old log files
            self.n,
            self.k,
            self.N,
            self.vote_count,
            self.transformation.__name__,
            self.combiner.__name__,
            self.learner.iterations,
            self.measured_time,
            accuracy,
            abortions,
            ','.join(map(str, self.model.weight_array.flatten() / norm(self.model.weight_array.flatten())))
        )
        self.progress_logger.info(msg)
        self.result_logger.info(msg)
