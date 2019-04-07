import re

from pypuf.experiments.experiment.property_test import ExperimentPropertyTest, Parameters
from pypuf.simulation.arbiter_based.ltfarray import CompoundTransformation, LTFArray
from pypuf.simulation.weak import NoisySRAM
from pypuf.studies.base import Study


class ReliabilityStudy(Study):

    def experiments(self):
        n = 64
        return [
            ExperimentPropertyTest(
                progress_log_name=None,
                parameters=Parameters(
                    test_function='reliability_statistic',
                    challenge_count=100,
                    measurements=20,
                    challenge_seed=31415,
                    ins_gen_function='create_noisy_ltf_arrays',
                    param_ins_gen={
                        'n': n,
                        'k': k,
                        'instance_count': 10,
                        'transformation': CompoundTransformation(
                            generator=LTFArray.generate_ipmod2_transform,
                            args=(n, k, NoisySRAM((k, n, n), seed=31415, noise=sram_noise)),
                            name='transform_ipmod2_NoisySRAM_31415_%f' % sram_noise
                        ),
                        'combiner': 'xor',
                        'bias': None,
                        'weight_random_seed': 31415,
                        'sigma_noise': arbiter_noise,
                    }
                )
            )
            for sram_noise in [.00001, .0001, .001, .01, .1, .2, .5]
            for arbiter_noise in [.00001, .0001, .001, .01, .1, .2, .5]
            for k in [1, 2, 4, 6, 8]
        ]

    def plot(self):
        data = self.experimenter.results
        data['sram_noise'] = data.apply(lambda row: re.sub(
            r'.*\'transformation\': transform_ipmod2_NoisySRAM_[0-9]+_([0-9\.]+).*',
            r'\1',
            row['param_ins_gen']
        ), axis=1)
        data['arbiter_noise'] = data.apply(lambda row: re.sub(
            r'.*\'sigma_noise\': ([^\,]+).*',
            r'\1',
            row['param_ins_gen']
        ), axis=1)
        self.experimenter.save_results()
