"""
Learn the lower XOR Arbiter PUF of an iPUF.
"""

from os import getpid
from typing import NamedTuple
from uuid import UUID
from numpy import logical_and, vstack, array, insert, abs as abs_np, sum as sum_np, expand_dims, absolute, sqrt, empty,\
    delete
from numpy.linalg import norm
from numpy.random.mtrand import RandomState
from scipy.special import erf
from scipy.stats import pearsonr

from pypuf.simulation.arbiter_based.ltfarray import NoisyLTFArray, LTFArray
from pypuf.tools import approx_dist, TrainingSet, random_inputs
from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.evolution_strategies.reliability_cmaes_learner import ReliabilityBasedCMAES
from pypuf.simulation.arbiter_based.arbiter_puf import InterposePUF


class Parameters(NamedTuple):
    n: int
    k_up: int
    k_down: int
    layer: str
    seed: int
    noisiness: float
    N: int
    R: int
    eps: float
    extra: int
    remove_error_1: bool
    remove_error_2: bool
    abort_delta: float
    max_tries: int
    gpu_id: int
    separate: bool
    heuristic: list


class Result(NamedTuple):
    experiment_id: UUID
    measured_time: float
    pid: int
    accuracy: float
    iterations: int
    stops: str
    max_possible_acc: float
    cross_correlation_lower: list
    cross_correlation_upper: list
    cross_correlation_rel_lower: list
    cross_correlation_rel_upper: list
    discard_count: dict
    iteration_count: dict
    fitness_histories: list
    fitnesses: list
    error_1: list
    error_2: list
    weights: list
    ts_ratios: list


class ExperimentCustomReliabilityBasedLayerIPUF(Experiment):
    """ This class implements an experiment for executing the reliability based CMAES learner for XOR LTF arrays.
    Furthermore, the learning results are being logged into csv files.
    """

    def __init__(self, progress_log_name, parameters: Parameters):
        """ Initialize an Experiment using the Reliability based CMAES Learner for modeling LTF Arrays.
        :param progress_log_name:   Log name, Prefix of the name of the experiment log file
        :param parameters:          Parameters object for this experiment
        """

        assert parameters.layer == 'lower' or parameters.layer == 'upper'
        super().__init__(
            progress_log_name=f'{progress_log_name}.0x{parameters.seed}_{parameters.k_up}_'
                              f'{parameters.k_down}_{parameters.n}_{parameters.N}_{parameters.R}',
            parameters=parameters,
        )
        self.prng = RandomState(seed=self.parameters.seed)
        self.simulation = None
        self.learner = None
        self.model = None
        self.ts = None
        self.learning_meta_data = None
        self.challenges = None
        self.responses = None
        self.s = None
        self.s_swap = None
        self.error_1 = []
        self.error_2 = []
        self.num_unreliable = 0
        self.num_reliable = 0
        self.layer = parameters.layer
        self.target_layer = None
        self.ts_ratios = []
        self.separate = parameters.separate
        self.heuristic = parameters.heuristic

    def generate_unreliable_challenges_for_layer(self):
        """Analysis outside of attacker model"""
        s = self.s if self.heuristic[0] else ~self.s
        s_swap = self.s_swap if self.heuristic[1] else ~self.s_swap
        idx_heuristic = logical_and(s, s_swap)
        self.num_unreliable = sum_np(idx_heuristic)
        cs_expand_unreliable = insert(self.challenges[idx_heuristic], self.simulation.interpose_pos, 1, axis=1)
        idx_layer_unrel = ~self.is_reliable(
            simulation=self.target_layer,
            challenges=cs_expand_unreliable if self.layer == 'lower' else self.challenges[idx_heuristic],
            repetitions=self.parameters.R,
            epsilon=self.parameters.eps,
        )
        if self.parameters.remove_error_1:
            self.num_unreliable = sum_np(idx_layer_unrel)
            cs_expand_unreliable = cs_expand_unreliable[idx_layer_unrel]
        num_unreliable = sum_np(idx_layer_unrel)
        self.ts_ratios += [num_unreliable, self.num_unreliable]
        self.error_1 = [1 - num_unreliable / self.num_unreliable]
        nums = [sum_np(~self.is_reliable(
            simulation=NoisyLTFArray(
                weight_array=expand_dims(self.simulation.down.weight_array[i, :-1], axis=0),
                combiner='xor',
                transform='atf',
                sigma_noise=self.simulation.down.sigma_noise,
                random_instance=self.simulation.down.random,
            ),
            challenges=cs_expand_unreliable,
            repetitions=self.parameters.R,
            epsilon=self.parameters.eps,
        )) for i in range(self.parameters.k_down)]
        nums += [sum_np(~self.is_reliable(
            simulation=NoisyLTFArray(
                weight_array=expand_dims(self.simulation.up.weight_array[i, :-1], axis=0),
                combiner='xor',
                transform='atf',
                sigma_noise=self.simulation.up.sigma_noise,
                random_instance=self.simulation.up.random,
            ),
            challenges=self.challenges[idx_heuristic][idx_layer_unrel],
            repetitions=self.parameters.R,
            epsilon=self.parameters.eps,
        )) for i in range(self.parameters.k_up)]
        self.error_1 += [1 - num_chain_unreliable / self.num_unreliable for num_chain_unreliable in nums]
        print(f'Out of {self.num_unreliable} chosen challenges, {num_unreliable} ({(1 - self.error_1[0]) * 100:.2f}%) '
              f'are actually unreliable on the {self.layer} layer.')
        return (self.challenges[idx_heuristic][idx_layer_unrel], self.responses[idx_heuristic][idx_layer_unrel]) \
            if self.parameters.remove_error_1 else (self.challenges[idx_heuristic], self.responses[idx_heuristic])

    def generate_reliable_challenges_for_layer(self):
        """Analysis outside of attacker model"""
        s = self.s if self.heuristic[2] else ~self.s
        s_swap = self.s_swap if self.heuristic[3] else ~self.s_swap
        idx_heuristic = logical_and(s, s_swap)
        self.num_reliable = sum_np(idx_heuristic)
        cs_expand_reliable = insert(self.challenges[idx_heuristic], self.simulation.interpose_pos, 1, axis=1)
        idx_layer_rel = self.is_reliable(
            simulation=self.target_layer,
            challenges=cs_expand_reliable if self.layer == 'lower' else self.challenges[idx_heuristic],
            repetitions=self.parameters.R,
            epsilon=self.parameters.eps,
        )
        if self.parameters.remove_error_2:
            self.num_reliable = sum_np(idx_layer_rel)
            cs_expand_reliable = cs_expand_reliable[idx_layer_rel]
        num_reliable = sum_np(idx_layer_rel)
        self.ts_ratios += [num_reliable, self.num_reliable]
        self.error_2 = [1 - num_reliable / self.num_reliable]
        nums = [sum_np(self.is_reliable(
            simulation=NoisyLTFArray(
                weight_array=expand_dims(self.simulation.down.weight_array[i, :-1], axis=0),
                combiner='xor',
                transform='atf',
                sigma_noise=self.simulation.down.sigma_noise,
                random_instance=self.simulation.down.random,
            ),
            challenges=cs_expand_reliable,
            repetitions=self.parameters.R,
            epsilon=self.parameters.eps,
        )) for i in range(self.parameters.k_down)]
        nums += [sum_np(self.is_reliable(
            simulation=NoisyLTFArray(
                weight_array=expand_dims(self.simulation.up.weight_array[i, :-1], axis=0),
                combiner='xor',
                transform='atf',
                sigma_noise=self.simulation.up.sigma_noise,
                random_instance=self.simulation.up.random,
            ),
            challenges=self.challenges[idx_heuristic][idx_layer_rel],
            repetitions=self.parameters.R,
            epsilon=self.parameters.eps,
        )) for i in range(self.parameters.k_up)]
        self.error_2 += [1 - num_chain_reliable / self.num_reliable for num_chain_reliable in nums]
        print(f'Out of {self.num_reliable} chosen challenges, {num_reliable} ({(1 - self.error_2[0]) * 100:.2f}%) '
              f'are actually reliable on the {self.layer} layer.')
        return (self.challenges[idx_heuristic][idx_layer_rel], self.responses[idx_heuristic][idx_layer_rel]) \
            if self.parameters.remove_error_2 else (self.challenges[idx_heuristic], self.responses[idx_heuristic])

    def generate_separate_tset(self):
        cs_train = random_inputs(self.parameters.n, self.ts.N, self.prng)
        cs_train_expand = insert(cs_train, self.simulation.interpose_pos, 1, axis=1)
        rs_train = array([self.target_layer.eval(challenges=cs_train_expand if self.layer == 'lower' else cs_train)
                          for _ in range(self.parameters.R)]).T
        idx_layer_rel = self.is_reliable(
            simulation=self.target_layer,
            challenges=cs_train_expand if self.layer == 'lower' else cs_train,
            repetitions=self.parameters.R,
            epsilon=self.parameters.eps,
        )
        num_reliable = sum_np(idx_layer_rel)
        num_unreliable = self.ts.N - num_reliable
        self.ts_ratios = [num_unreliable, num_unreliable, num_reliable, num_reliable]
        self.error_1 = [0]
        self.error_2 = [0]
        nums_unreliable = [sum_np(~self.is_reliable(
            simulation=NoisyLTFArray(
                weight_array=expand_dims(self.simulation.down.weight_array[i, :-1], axis=0),
                combiner='xor',
                transform='atf',
                sigma_noise=self.simulation.down.sigma_noise,
                random_instance=self.simulation.down.random,
            ),
            challenges=cs_train_expand[~idx_layer_rel],
            repetitions=self.parameters.R,
            epsilon=self.parameters.eps,
        )) for i in range(self.parameters.k_down)]
        nums_unreliable += [sum_np(~self.is_reliable(
            simulation=NoisyLTFArray(
                weight_array=expand_dims(self.simulation.up.weight_array[i, :-1], axis=0),
                combiner='xor',
                transform='atf',
                sigma_noise=self.simulation.up.sigma_noise,
                random_instance=self.simulation.up.random,
            ),
            challenges=cs_train[~idx_layer_rel],
            repetitions=self.parameters.R,
            epsilon=self.parameters.eps,
        )) for i in range(self.parameters.k_up)]
        self.error_2 += [1 - num_chain_unreliable / num_unreliable for num_chain_unreliable in nums_unreliable]
        nums_reliable = [sum_np(self.is_reliable(
            simulation=NoisyLTFArray(
                weight_array=expand_dims(self.simulation.down.weight_array[i, :-1], axis=0),
                combiner='xor',
                transform='atf',
                sigma_noise=self.simulation.down.sigma_noise,
                random_instance=self.simulation.down.random,
            ),
            challenges=cs_train_expand[idx_layer_rel],
            repetitions=self.parameters.R,
            epsilon=self.parameters.eps,
        )) for i in range(self.parameters.k_down)]
        nums_reliable += [sum_np(self.is_reliable(
            simulation=NoisyLTFArray(
                weight_array=expand_dims(self.simulation.up.weight_array[i, :-1], axis=0),
                combiner='xor',
                transform='atf',
                sigma_noise=self.simulation.up.sigma_noise,
                random_instance=self.simulation.up.random,
            ),
            challenges=cs_train[idx_layer_rel],
            repetitions=self.parameters.R,
            epsilon=self.parameters.eps,
        )) for i in range(self.parameters.k_up)]
        self.error_1 += [1 - num_chain_reliable / num_reliable for num_chain_reliable in nums_reliable]

        return cs_train_expand if self.layer == 'lower' else cs_train, rs_train

    def run(self):
        """ Initialize the instance, the training set and the learner to then run the Reliability based CMAES with the
        given parameters.
        """
        # Instantiate the baseline noisy iPUF from which the lower chains shall be learned
        self.simulation = InterposePUF(
            n=self.parameters.n,
            k_up=self.parameters.k_up,
            k_down=self.parameters.k_down,
            transform='atf',
            seed=self.prng.randint(2 ** 32),
            noisiness=self.parameters.noisiness,
            noise_seed=self.prng.randint(2 ** 32),
        )
        self.target_layer = self.simulation.down if self.layer == 'lower' else self.simulation.up
        self.ts = TrainingSet(self.simulation, 1, self.prng, self.parameters.R)

        self.generate_crp_set(self.parameters.N)
        # Build training Set for learning the lower chains of the IPUF
        cs_unreliable, rs_unreliable = self.generate_unreliable_challenges_for_layer()
        cs_reliable, rs_reliable = self.generate_reliable_challenges_for_layer()
        cs_train = vstack((cs_unreliable, cs_reliable))
        rs_train = vstack((rs_unreliable, rs_reliable))
        if self.layer == 'lower':
            cs_train = insert(cs_train, self.simulation.interpose_pos, 1, axis=1)
            # cs_train_p = insert(cs_train, self.simulation.interpose_pos, 1, axis=1)
            # cs_train_m = insert(cs_train, self.simulation.interpose_pos, -1, axis=1)
            # cs_train = vstack((cs_train_p, cs_train_m))
            # rs_train = vstack((rs_train, rs_train))
        self.ts.instance = self.simulation
        self.ts.N = cs_train.shape[0]
        if self.separate:
            cs_train, rs_train = self.generate_separate_tset()
        self.ts.challenges = cs_train
        self.ts.responses = rs_train
        print(f'Generated Training Set: Reliables: {cs_reliable.shape[0]}\n'
              f'Unreliables ({self.layer}): {cs_unreliable.shape[0]} TrainSetSize: {cs_train.shape[0]}')

        # Instantiate the CMA-ES learner
        self.learner = ReliabilityBasedCMAES(
            training_set=self.ts,
            k=self.target_layer.k + self.parameters.extra,   # find more chains for analysis purposes
            n=self.target_layer.n,
            transform=self.target_layer.transform,
            combiner=self.target_layer.combiner,
            abort_delta=self.parameters.abort_delta,
            random_seed=self.prng.randint(2 ** 32),
            max_tries=self.parameters.max_tries,
            logger=self.progress_logger,
            gpu_id=self.gpu_id,
        )

        # Start learning a model
        self.model, self.learning_meta_data = self.learner.learn()

    @staticmethod
    def normalize(weights):
        return weights / norm(weights)

    def analyze(self):
        """
            Analyze the results and return the Results object.
        """
        n = self.parameters.n + 1 + 1

        # Accuracy of the learned model using 10000 random samples
        empirical_accuracy = 1 - approx_dist(self.target_layer, self.model, 10000, self.prng)

        # Accuracy of the base line Noisy LTF. Can be < 1.0 since it is noisy
        best_empirical_accuracy = 1 - approx_dist(self.target_layer, self.target_layer, 10000, self.prng)
        # Correl. of the learned model and the base line LTF using pearson for all chains
        cross_correlation_lower = [[round(pearsonr(
            v,
            w[array(range(n)) != ((n - 2) // 2)] if self.layer == 'upper' else w,
        )[0], 2)
                                    for w in self.ts.instance.down.weight_array]
                                   for v in self.model.weight_array]
        cross_correlation_upper = [[round(pearsonr(
            v[array(range(n)) != ((n - 2) // 2)] if self.layer == 'lower' else v,
            w,
        )[0], 2)
                                    for w in self.ts.instance.up.weight_array]
                                   for v in self.model.weight_array]
        # Correlation of learned model and target LTF using reliability analysis
        model_chains = [LTFArray(
            weight_array=self.normalize(self.model.weight_array[i:i + 1, :-1]),
            transform=self.model.transform,
            combiner=self.model.combiner,
        ) for i in range(self.model.weight_array.shape[0])]
        target_chains_lower = [LTFArray(
            weight_array=self.normalize(self.simulation.down.weight_array[i:i + 1, :-1] if self.layer == 'lower'
                                        else self.simulation.down.weight_array[
                                             i:i + 1, array(range(n)) != ((n - 2) // 2)][:, :-1]),
            transform=self.simulation.down.transform,
            combiner=self.simulation.down.combiner,
        ) for i in range(self.parameters.k_down)]
        target_chains_upper = [LTFArray(
            weight_array=self.normalize(self.simulation.up.weight_array[i:i + 1, :-1]),
            transform=self.simulation.up.transform,
            combiner=self.simulation.up.combiner,
        ) for i in range(self.parameters.k_up)]
        cross_correlation_rel_lower = empty((self.parameters.k_down + self.parameters.extra, self.parameters.k_down))
        cross_correlation_rel_upper = empty((self.parameters.k_down + self.parameters.extra, self.parameters.k_up))
        for i, chain in enumerate(model_chains):
            chain_reliabilities = absolute(chain.val(self.ts.challenges))
            theoretical_stab = erf(chain_reliabilities / sqrt(2) / self.parameters.noisiness)
            for j, target_chain in enumerate(target_chains_lower):
                target_chain_reliabilities = absolute(target_chain.val(self.ts.challenges))
                target_theoretical_stab = erf(target_chain_reliabilities / sqrt(2) / self.parameters.noisiness)
                cross_correlation_rel_lower[i, j] = round(pearsonr(target_theoretical_stab, theoretical_stab)[0], 2)
            for k, target_chain in enumerate(target_chains_upper):
                target_chain_reliabilities = absolute(target_chain.val(
                    delete(self.ts.challenges, self.simulation.interpose_pos, axis=1) if self.layer == 'lower'
                    else self.ts.challenges
                ))
                target_theoretical_stab = erf(target_chain_reliabilities / sqrt(2) / self.parameters.noisiness)
                cross_correlation_rel_upper[i, k] = round(pearsonr(target_theoretical_stab, theoretical_stab)[0], 2)

        return Result(
            experiment_id=self.id,
            measured_time=self.measured_time,
            pid=getpid(),
            accuracy=empirical_accuracy,
            iterations=self.learner.num_tries,
            stops=self.learner.stops,
            max_possible_acc=best_empirical_accuracy,
            cross_correlation_lower=cross_correlation_lower,
            cross_correlation_upper=cross_correlation_upper,
            cross_correlation_rel_lower=cross_correlation_rel_lower.tolist(),
            cross_correlation_rel_upper=cross_correlation_rel_upper.tolist(),
            discard_count=self.learning_meta_data['discard_count'],
            iteration_count=self.learning_meta_data['iteration_count'],
            fitness_histories=self.learning_meta_data['fitness_histories'],
            fitnesses=[histories[-1] for histories in self.learning_meta_data['fitness_histories']],
            error_1=self.error_1,
            error_2=self.error_2,
            weights=self.model.weight_array,
            ts_ratios=self.ts_ratios,
        )

    def interpose(self, challenges, bit):
        return insert(challenges, self.simulation.interpose_pos, bit, axis=1)

    @staticmethod
    def is_reliable(simulation, challenges, repetitions=11, epsilon=0.9):
        responses = array([simulation.eval(challenges=challenges) for _ in range(repetitions)])
        axis = 0
        return abs_np(sum_np(responses, axis=axis)) / (2 * responses.shape[axis]) + 0.5 >= epsilon

    def generate_crp_set(self, N):
        cs = random_inputs(self.parameters.n, N, self.prng)
        cs_swap = cs.copy()
        cs_swap[:, self.simulation.interpose_pos // 2 - 1] *= -1
        cs_swap[:, self.simulation.interpose_pos // 2] *= -1

        s = self.is_reliable(
            simulation=self.simulation,
            challenges=cs,
            repetitions=self.parameters.R,
            epsilon=self.parameters.eps,
        )
        s_swap = self.is_reliable(
            simulation=self.simulation,
            challenges=cs_swap,
            repetitions=self.parameters.R,
            epsilon=self.parameters.eps,
        )
        l_p = self.is_reliable(
            simulation=self.simulation.down,
            challenges=self.interpose(cs, 1),
            repetitions=self.parameters.R,
            epsilon=self.parameters.eps,
        )
        l_p_swap = self.is_reliable(
            simulation=self.simulation.down,
            challenges=self.interpose(cs_swap, 1),
            repetitions=self.parameters.R,
            epsilon=self.parameters.eps,
        )
        l_m = self.is_reliable(
            simulation=self.simulation.down,
            challenges=self.interpose(cs, -1),
            repetitions=self.parameters.R,
            epsilon=self.parameters.eps,
        )
        l_m_swap = self.is_reliable(
            simulation=self.simulation.down,
            challenges=self.interpose(cs_swap, -1),
            repetitions=self.parameters.R,
            epsilon=self.parameters.eps,
        )

        for label, event, condition in [
            ('l_p|s', l_p, s),
            ('l_p|s_swap', l_p, s_swap),
            ('l_p|11', l_p, s & s_swap),
            ('l_p|10', l_p, s & ~s_swap),
            ('l_p|01', l_p, ~s & s_swap),
            ('l_p|00', l_p, ~s & ~s_swap),
            ('l_p_swap|s', l_p_swap, s),
            ('l_p_swap|s_swap', l_p_swap, s_swap),
            ('l_p_swap|11', l_p_swap, s & s_swap),
            ('l_p_swap|10', l_p_swap, s & ~s_swap),
            ('l_p_swap|01', l_p_swap, ~s & s_swap),
            ('l_p_swap|00', l_p_swap, ~s & ~s_swap),
            ('l_m|s', l_m, s),
            ('l_m|s_swap', l_m, s_swap),
            ('l_m|11', l_m, s & s_swap),
            ('l_m|10', l_m, s & ~s_swap),
            ('l_m|01', l_m, ~s & s_swap),
            ('l_m|00', l_m, ~s & ~s_swap),
            ('l_m_swap|s', l_m_swap, s),
            ('l_m_swap|s_swap', l_m_swap, s_swap),
            ('l_m_swap|11', l_m_swap, s & s_swap),
            ('l_m_swap|10', l_m_swap, s & ~s_swap),
            ('l_m_swap|01', l_m_swap, ~s & s_swap),
            ('l_m_swap|00', l_m_swap, ~s & ~s_swap),
        ]:
            count_condition = sum_np(condition)
            coincidences = sum_np(logical_and(event, condition))
            print(f'{label}: {coincidences/count_condition:.4f} (total: {coincidences})')

        self.challenges = cs
        self.responses = array([self.simulation.eval(challenges=cs) for _ in range(self.parameters.R)]).T
        self.s = s
        self.s_swap = s_swap
