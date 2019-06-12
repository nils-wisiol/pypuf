import re

from seaborn import catplot

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
                    measurements=111,
                    challenge_seed=31415,
                    ins_gen_function='create_noisy_ltf_arrays',
                    param_ins_gen={
                        'n': n,
                        'k': k,
                        'instance_count': 10,
                        'transformation': CompoundTransformation(
                            generator=LTFArray.generate_bent_transform,
                            args=(n, k, NoisySRAM(n, noise=sram_noise, seed_skew=31415, seed_noise=314)),
                            name='transform_bent_NoisySRAM_31415_%f' % sram_noise
                        ),
                        'combiner': 'xor',
                        'bias': None,
                        'weight_random_seed': 31415,
                        'sigma_noise': arbiter_noise,
                    }
                )
            )
            for sram_noise in [0, .05, .1, .15, .2, .25, .4, .5]
            for arbiter_noise in [0, 0.02, 0.05, 0.1, 0.15, .3, .4, .5]
            for k in [1]  #, 2, 4, 6, 8]
        ]

    def plot(self):
        data = self.experimenter.results
        data['sram_noise'] = data.apply(lambda row: re.sub(
            r'transform_bent_NoisySRAM_[0-9]+_([0-9.]+).*',
            r'\1',
            row['param_ins_gen__transformation']
        ), axis=1)

        facet = catplot(
            x='sram_noise',
            y='mean',
            col='param_ins_gen__k',
            hue='param_ins_gen__sigma_noise',
            kind='bar',
            data=data,
        )

        facet.set_axis_labels('SRAM Noise Level', 'Reliability')
        facet.fig.subplots_adjust(top=.8, wspace=.02, hspace=.02);
        facet.fig.suptitle('Bent XOR Arbiter PUF Reliabilty')
        facet.fig.savefig('figures/%s.pdf' % self.name(), bbox_inches='tight', pad_inches=.5)
