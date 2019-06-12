from os import getpid
from typing import NamedTuple, List
from uuid import UUID

from numpy import broadcast_to, sign, sum, mean, zeros, absolute
from numpy.random.mtrand import RandomState
from seaborn import catplot, axes_style

from pypuf.experiments.experiment.base import Experiment
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, NoisyLTFArray, CompoundTransformation
from pypuf.simulation.weak import NoisySRAM
from pypuf.studies.base import Study
from pypuf.tools import sample_inputs


class ReliabilityExperimentResult(NamedTuple):
    experiment_id: UUID
    pid: int
    measured_time: float
    mean_bit_error_rate: float


class NoisySRAMReliabilityExperimentParameters(NamedTuple):
    N: int  # number of different challenges to use
    R: int  # number of repeated measurements of a each challenge
    I: int  # number of instances
    seed_challenges: int
    noisy_sram_size: int
    noisy_sram_noise: float
    noisy_sram_seed_skew: int
    noisy_sram_seed_noise: int


class NoisyLTFArrayReliabilityExperimentParameters(NamedTuple):
    N: int  # number of different challenges to use
    R: int  # number of repeated measurements of a each challenge
    I: int  # number of instances
    seed_challenges: int
    noisy_ltfarray_n: int
    noisy_ltfarray_k: int
    noisy_ltfarray_seed_weights: int
    noisy_ltfarray_seed_noise: int
    noisy_ltfarray_noisiness: float
    noisy_ltfarray_transform: str
    noisy_ltfarray_combiner: str


class NoisyIPMod2TransformLTFArrayReliabilityExperimentParameters(NamedTuple):
    N: int  # number of different challenges to use
    R: int  # number of repeated measurements of a each challenge
    I: int  # number of instances
    ipmod2_length: int
    seed_challenges: int
    noisy_ltfarray_n: int
    noisy_ltfarray_k: int
    noisy_ltfarray_seed_weights: int
    noisy_ltfarray_seed_noise: int
    noisy_ltfarray_noisiness: float
    noisy_sram_noise: float
    noisy_sram_seed_skew: int
    noisy_sram_seed_noise: int


class ReliabilityExperiment(Experiment):

    def __init__(self, progress_log_name, parameters):
        super().__init__(progress_log_name, parameters)
        self.parameters = parameters
        self.instance_bit_error_rates = None
        self.instances = None

    def run(self):
        (R, N) = (self.parameters.R, self.parameters.N)
        instances = self.instances
        I = len(instances)
        (n, m) = (instances[0].challenge_length(), instances[0].response_length())
        self.instance_bit_error_rates = zeros((I, m))
        for i, instance in enumerate(instances):
            assert instance.response_length() == m
            challenges = sample_inputs(instance.challenge_length(), N,
                                       random_instance=RandomState(seed=self.parameters.seed_challenges))
            for idx, c in enumerate(challenges):
                # compute R responses to the same challenge
                assert c.shape == (n, )
                repeated_c = broadcast_to(c, (R, n))
                assert repeated_c.shape == (R, n)
                responses = instance.eval(repeated_c).reshape((R, m))  # shape (R, m)
                assert responses.shape == (R, m), 'responses expected to have shape %s, but had %s' % ((R, m), responses.shape)

                # determine the 'correct' response by majority vote over all R responses
                correct_response = sign(sum(responses, axis=0))  # shape (m, )
                assert correct_response.shape == (m, )

                # compute the average deviation from the correct response
                all_correct_responses = broadcast_to(correct_response, (R, m))  # shape (R, m)
                assert all_correct_responses.shape == (R, m)
                mean_bit_error_rate = mean(absolute(all_correct_responses - responses), axis=0) / 2  # shape (m, )
                assert mean_bit_error_rate.shape == (m, )
                self.instance_bit_error_rates[i] += mean_bit_error_rate

    def analyze(self):
        return ReliabilityExperimentResult(
            experiment_id=self.id,
            pid=getpid(),
            measured_time=self.measured_time,
            mean_bit_error_rate=mean(self.instance_bit_error_rates) / self.parameters.N,
        )


class NoisySRAMReliabilityExperiment(ReliabilityExperiment):

    def prepare(self):
        self.instances = [
            NoisySRAM(
                size=self.parameters.noisy_sram_size,
                noise=self.parameters.noisy_sram_noise,
                seed_skew=self.parameters.noisy_sram_seed_skew + i,
                seed_noise=(self.parameters.noisy_sram_seed_noise * i) % 2**32,
            ) for i in range(self.parameters.I)
        ]


class NoisyLTFArrayReliabilityExperiment(ReliabilityExperiment):

    def prepare(self):
        self.instances = [
            NoisyLTFArray(
                weight_array=LTFArray.normal_weights(
                    self.parameters.noisy_ltfarray_n, self.parameters.noisy_ltfarray_k,
                    random_instance=RandomState(seed=self.parameters.noisy_ltfarray_seed_weights + i),
                ),
                transform=self.parameters.noisy_ltfarray_transform,
                combiner=self.parameters.noisy_ltfarray_combiner,
                sigma_noise=NoisyLTFArray.sigma_noise_from_random_weights(
                    n=self.parameters.noisy_ltfarray_n,
                    sigma_weight=1,
                    noisiness=self.parameters.noisy_ltfarray_noisiness,
                ),
                random_instance=RandomState(seed=self.parameters.noisy_ltfarray_seed_noise)
            ) for i in range(self.parameters.I)
        ]


class NoisyIPMod2TransformLTFArrayReliabilityExperiment(ReliabilityExperiment):

    def prepare(self):
        self.instances = [
            NoisyLTFArray(
                weight_array=LTFArray.normal_weights(
                    self.parameters.noisy_ltfarray_n, self.parameters.noisy_ltfarray_k,
                    random_instance=RandomState(seed=self.parameters.noisy_ltfarray_seed_weights + i),
                ),
                transform=LTFArray.generate_bent_transform(
                    n=self.parameters.noisy_ltfarray_n,
                    kk=self.parameters.noisy_ltfarray_k,
                    weak_puf=NoisySRAM(
                        size=self.parameters.noisy_ltfarray_k * self.parameters.noisy_ltfarray_n *
                             self.parameters.ipmod2_length,
                        noise=self.parameters.noisy_sram_noise,
                        seed_skew=self.parameters.noisy_sram_seed_skew + i,
                        seed_noise=self.parameters.noisy_sram_seed_noise + i,
                    ),
                    ipmod2_length=self.parameters.ipmod2_length,
                ),
                combiner='xor',
                sigma_noise=NoisyLTFArray.sigma_noise_from_random_weights(
                    n=self.parameters.noisy_ltfarray_n,
                    sigma_weight=1,
                    noisiness=self.parameters.noisy_ltfarray_noisiness,
                ),
                random_instance=RandomState(seed=self.parameters.noisy_ltfarray_seed_noise)
            ) for i in range(self.parameters.I)
        ]


class ReliabilityStudy(Study):

    def experiments(self) -> List[Experiment]:
        return \
            [
                NoisySRAMReliabilityExperiment(
                    progress_log_name=None,
                    parameters=NoisySRAMReliabilityExperimentParameters(
                        N=1,
                        R=551,
                        I=1000,
                        seed_challenges=31415,
                        noisy_sram_size=size,
                        noisy_sram_noise=noise,
                        noisy_sram_seed_skew=31415,
                        noisy_sram_seed_noise=0xdeadbeef,
                    )
                )
                for noise in [0, .001, .01, .02, .03, .04, .05, .1, .2, .5, 1, 2, 5, 100]
                for size in [1024, 2048]
            ] + [
                NoisyLTFArrayReliabilityExperiment(
                    progress_log_name=None,
                    parameters=NoisyLTFArrayReliabilityExperimentParameters(
                        N=500,
                        R=151,
                        I=100,
                        seed_challenges=31415,
                        noisy_ltfarray_n=64,
                        noisy_ltfarray_k=k,
                        noisy_ltfarray_seed_weights=31415,
                        noisy_ltfarray_seed_noise=271828,
                        noisy_ltfarray_noisiness=noisiness,
                        noisy_ltfarray_transform='atf',
                        noisy_ltfarray_combiner='xor',
                    )
                )
                for noisiness in [.01, .02, .05, .1, .2]
                for k in [1, 2, 4, 8, 12]
            ] + [
                NoisyIPMod2TransformLTFArrayReliabilityExperiment(
                    progress_log_name=None,
                    parameters=NoisyIPMod2TransformLTFArrayReliabilityExperimentParameters(
                        N=500,
                        R=151,
                        I=100,
                        ipmod2_length=ipmod2_length,
                        seed_challenges=31415,
                        noisy_ltfarray_n=64,
                        noisy_ltfarray_k=k,
                        noisy_ltfarray_seed_weights=31415,
                        noisy_ltfarray_seed_noise=271828,
                        noisy_ltfarray_noisiness=ltfarray_nosiness,
                        noisy_sram_noise=sram_noise,
                        noisy_sram_seed_skew=27182,
                        noisy_sram_seed_noise=314159,
                    )
                )
                for ltfarray_nosiness in [.01, .02]  # , .05, .1, .2]
                for sram_noise in [.05, .1]  # , .2, .3, .4, 1, 10, 100]
                for k in [1]  # , 2, 4, 8, 12]
                for ipmod2_length in [8, 16, 32, 64]
            ]

    def plot(self):
        data = self.experimenter.results

        with axes_style("whitegrid"):

            plot_data = data[data['experiment'] == 'NoisySRAMReliabilityExperiment']
            if not plot_data.empty:
                facet = catplot(
                    x='noisy_sram_noise',
                    y='mean_bit_error_rate',
                    data=plot_data,
                    kind='bar',
                )

                facet.set_axis_labels('SRAM Noise Level', 'Mean Bit Error Rate')
                facet.fig.set_size_inches(12, 4)
                facet.fig.subplots_adjust(top=.8, wspace=.02, hspace=.02)
                facet.fig.suptitle('Noisy PUF Mean Bit Error Rate')
                facet.fig.savefig('figures/%s.sram.pdf' % self.name(), bbox_inches='tight', pad_inches=.5)

            plot_data = data[data['experiment'] == 'NoisyLTFArrayReliabilityExperiment']
            if not plot_data.empty:
                facet = catplot(
                    x='noisy_ltfarray_noisiness',
                    y='mean_bit_error_rate',
                    hue='noisy_ltfarray_k',
                    data=plot_data,
                    kind='bar',
                )

                facet.set_axis_labels('LTFArray Noisiness Level', 'Mean Bit Error Rate')
                facet.fig.set_size_inches(12, 4)
                facet.fig.subplots_adjust(top=.8, wspace=.02, hspace=.02)
                facet.fig.suptitle('Noisy PUF Mean Bit Error Rate')
                facet.fig.savefig('figures/%s.ltfarray.pdf' % self.name(), bbox_inches='tight', pad_inches=.5)

            plot_data = data[data['experiment'] == 'NoisyIPMod2TransformLTFArrayReliabilityExperiment']
            if not plot_data.empty:
                facet = catplot(
                    x='noisy_ltfarray_noisiness',
                    y='mean_bit_error_rate',
                    col='noisy_ltfarray_k',
                    hue='noisy_sram_noise',
                    kind='bar',
                    data=plot_data,
                )

                facet.set_axis_labels('SRAM Noise Level', 'Mean Bit Error Rate')
                facet.fig.set_size_inches(12, 4)
                facet.fig.subplots_adjust(top=.8, wspace=.02, hspace=.02)
                facet.fig.suptitle('IPMod2 Input Transform XOR Arbiter PUF')
                facet.fig.savefig('figures/%s.ipmod2.pdf' % self.name(), bbox_inches='tight', pad_inches=.5)
