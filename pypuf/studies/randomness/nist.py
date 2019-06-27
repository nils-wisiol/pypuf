from os import getpid
from typing import NamedTuple
from uuid import UUID

import chaospy as chaospy
import sp80022suite
from numpy import append, array, int8, around, flip, nan, nanmean, empty
from numpy.core.multiarray import ndarray
from numpy.random.mtrand import RandomState
from seaborn import catplot

from pypuf.experiments.experiment.base import Experiment
from pypuf.simulation.arbiter_based.arbiter_puf import XORArbiterPUF, LightweightSecurePUF, InterposePUF
from pypuf.simulation.base import Simulation
from pypuf.studies.base import Study
from pypuf.tools import vdc_sequence_bits, random_inputs


class Parameters(NamedTuple):
    simulation: Simulation
    n: int
    k: int
    sequencer: str
    seed_challenge: list
    length: int


class Result(NamedTuple):
    experiment_id: UUID
    pid: int
    measured_time: float
    sequence_head: str
    sequence_tail: str
    frequency_p: float = None
    block_frequency_p: float = None
    runs_p: float = None
    longest_run_of_ones_p: float = None
    rank_p: float = None
    discrete_fourier_transform_p: float = None
    non_overlapping_template_matchings_p: float = None
    overlapping_template_matchings_p: float = None
    universal_p: float = None
    linear_complexity_p: float = None
    approximate_entropy_p: float = None
    serial_p: float = None
    cumulative_sums_p: float = None
    random_excursions_p: float = None
    random_excursions_variant_p: float = None


class NISTRandomnessExperiment(Experiment):

    TESTS = [
        [sp80022suite.frequency, []],
        [sp80022suite.block_frequency, [10]],
        [sp80022suite.runs, []],
        [sp80022suite.longest_run_of_ones, []],
        [sp80022suite.rank, []],
        [sp80022suite.discrete_fourier_transform, []],
        [sp80022suite.non_overlapping_template_matchings, [4]],
        [sp80022suite.overlapping_template_matchings, [9]],
        [sp80022suite.universal, []],
        [sp80022suite.linear_complexity, [1000]],
        [sp80022suite.approximate_entropy, [6]],
        [sp80022suite.serial, [6]],
        [sp80022suite.cumulative_sums, []],
        [sp80022suite.random_excursions, []],
        [sp80022suite.random_excursions_variant, []],
    ]

    def __init__(self, progress_log_name, parameters: Parameters):
        super().__init__(progress_log_name, parameters)
        self.results = {}
        try:
            self.sequencer = getattr(self, 'sequencer_%s' % parameters.sequencer)
        except AttributeError:
            raise ValueError('Unknown sequencer %s.' % parameters.sequencer)

    @staticmethod
    def _sequencer_feedback_shift(instance: Simulation, seed: list, length: int, feedback_method: callable):
        state = 2 * array(seed.copy(), dtype=int8) - 1
        while len(state) < length + len(seed):
            state = append(state, instance.eval(feedback_method(state)))
        return .5 - .5 * state[-length:]

    @classmethod
    def sequencer_feedback_shift_tail(cls, instance: Simulation, seed: list, length: int) -> ndarray:
        n = instance.challenge_length()
        return cls._sequencer_feedback_shift(instance, seed, length, lambda state: state[-n:].reshape(1, n))

    @classmethod
    def sequencer_feedback_shift_head(cls, instance: Simulation, seed: list, length: int) -> ndarray:
        n = instance.challenge_length()
        return cls._sequencer_feedback_shift(instance, seed, length, lambda state: flip(state[-n:]).reshape(1, n))

    @classmethod
    def sequencer_round_robin_xor(cls, instance: Simulation, seed: list, length: int) -> ndarray:
        pointer = 0
        state = 2 * array([seed[:]]) - 1
        sequence = empty(shape=(length,))
        _, n = state.shape
        assert n == instance.challenge_length()
        for idx in range(length):
            # evaluate the state
            r = instance.eval(state)
            sequence[idx] = r
            # for even idx: XOR response with state bit under pointer
            # for odd idx: XOR negated response with state bit under pointer
            state[0, pointer] = state[0, pointer] * r * (1 if idx % 2 == 0 else -1)
            # move pointer
            pointer = (pointer + 1) % n
        return .5 - .5 * sequence

    @staticmethod
    def sequencer_vdc(instance: Simulation, seed: list, length: int) -> ndarray:
        n = instance.challenge_length()
        assert seed == [0] * n, "Seed must be [0,...,0] for Van der Corput sequencer."
        challenges = []
        for i in range(length):
            challenges.append(vdc_sequence_bits(i + 1, instance.challenge_length()))
        challenges = 2 * array(challenges, dtype=int8) - 1
        assert challenges.shape == (length, n)
        return .5 - .5 * instance.eval(challenges)

    @staticmethod
    def sequencer_halton(instance: Simulation, seed: list, length: int) -> ndarray:
        n = instance.challenge_length()
        assert seed == [0] * n, "Seed must be [0,...,0] for Halton sequencer."
        challenges = 2 * array(around(chaospy.create_halton_samples(length, dim=n)), dtype=int8).T - 1
        assert challenges.shape == (length, n)
        return .5 - .5 * instance.eval(challenges)

    @staticmethod
    def sequencer_random(instance: Simulation, seed: list, length: int) -> ndarray:
        seed = int(hash(''.join(map(str, seed)))) % 2**31
        return .5 - .5 * instance.eval(random_inputs(instance.challenge_length(), length, RandomState(seed=seed)))

    def run(self):
        sequence = self.sequencer(self.parameters.simulation, self.parameters.seed_challenge, self.parameters.length)
        self.results['sequence_head'] = '"' + ''.join([str(int(b)) for b in sequence[:256]]) + '"'
        self.results['sequence_tail'] = '"' + ''.join([str(int(b)) for b in sequence[-256:]]) + '"'
        byte_sequence = bytes(array(sequence, dtype=int8))
        for test, args in self.TESTS:
            try:
                result = test(*args, byte_sequence)
            except ValueError as e:
                result = str(e)

            self.results['%s_p' % test.__name__] = result

    def analyze(self):
        return Result(
            experiment_id=self.id,
            pid=getpid(),
            measured_time=self.measured_time,
            **self.results
        )


class NISTTest(Study):
    SAMPLE_SIZE = 100
    SHUFFLE = True
    SIMULATION_CLASSES = [XORArbiterPUF, LightweightSecurePUF, InterposePUF]

    def experiments(self):
        return [
            NISTRandomnessExperiment(
                progress_log_name=None,
                parameters=Parameters(
                    simulation=simulation_class(n=n, k=k, seed=i),
                    n=n,
                    k=k,
                    sequencer=sequencer,
                    seed_challenge=seed,
                    length=500000,
                )
            )
            for i in range(self.SAMPLE_SIZE)
            for n in [64, 128, 256]
            for k in [1, 2, 3, 4, 5, 6, 7, 8]
            for sequencer, seeds in [
                ('feedback_shift_head', [[0] * n, [0, 1] * (n // 2), [1, 0] + [1] * (n - 2)]),
                ('feedback_shift_tail', [[0] * n, [0, 1] * (n // 2), [1, 0] + [1] * (n - 2)]),
                ('random', [[0] * n]),
                ('halton', [[0] * n]),
                ('round_robin_xor', [[0] * n]),
            ]
            for seed in seeds
            for simulation_class in self.SIMULATION_CLASSES
        ]

    def plot(self):
        # original_data = self.experimenter.results
        # data = DataFrame()
        # for test, _ in NISTRandomnessExperiment.TESTS:
        #     data_copy = original_data.copy()
        #     data_copy['p_value'] = data_copy.apply(
        #         lambda row: row['%s_p' % test.__name__] if isinstance(row['%s_p' % test.__name__], float) else NaN,
        #         axis=1
        #     )
        #     data_copy['test'] = data_copy.apply(lambda _: test.__name__, axis=1)
        #     data = data.append(data_copy)
        #
        # data['simulation_name'] = data.apply(
        #     lambda row: re.match(
        #         '<(?P<name>[A-Za-z0-9._]+) object at 0x[0-9a-z]+>', str(row['simulation']))['name'].split('.')[-1],
        #     axis=1
        # )
        #
        # kwargs = dict(
        #     hue='k',
        #     y='p_value',
        #     x='simulation_name',
        #     col='sequencer',
        #     legend='full',
        # )
        #
        # facet_grid = relplot(
        #     style='n',
        #     row='test',
        #     data=data,
        #     kind="line",
        #     err_kws={'alpha': .03},
        #     **kwargs,
        # )
        # facet_grid.savefig("figures/%s.pdf" % self.name())

        data = self.experimenter.results
        data['avg_score'] = data.apply(
            lambda row: nanmean([
                row[f'{test_function.__name__}_p'] if isinstance(row[f'{test_function.__name__}_p'], float) else nan
                for (test_function, _) in NISTRandomnessExperiment.TESTS
            ]),
            axis=1,
        )

        def challenge_type(challenge):
            if challenge.startswith('[0, 1'):
                return '[01...01]'
            elif challenge.startswith('[0, 0'):
                return '[00...00]'
            elif challenge.startswith('[1, 0, 1'):
                return '[101...1]'
            else:
                raise ValueError(f'Unknown challenge type for challenge {challenge}.')

        data['seed_pattern'] = data.apply(
            lambda row: challenge_type(str(row['seed_challenge'])),
            axis=1,
        )

        facet_grid = catplot(
            data=data[(data['sequencer'] != 'halton') & (data['sequencer'] != 'linear_feedback_shift')],
            x='k',
            y='avg_score',
            col='sequencer',
            row='seed_pattern',
            hue='n',
            legend='full',
            kind='violin',  # try 'swarm', 'violin'
        )
        facet_grid.fig.set_size_inches(30, 20)
        facet_grid.savefig("figures/%s-avg.pdf" % self.name())
