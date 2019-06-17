import re

from seaborn import catplot

from pypuf.experiments.experiment.property_test import ExperimentPropertyTest, Parameters
from pypuf.simulation.arbiter_based.ltfarray import CompoundTransformation, LTFArray
from pypuf.simulation.weak import MajorityVoteNoisySRAM
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
                    measurements=11,
                    challenge_seed=31415,
                    ins_gen_function='create_noisy_ltf_arrays',
                    param_ins_gen={
                        'n': n,
                        'k': k,
                        'instance_count': 10,
                        'transformation': CompoundTransformation(
                            generator=LTFArray.generate_bent_transform,
                            args=(n, k, MajorityVoteNoisySRAM(n, sram_noise, votes, seed_skew=31415, seed_noise=314)),
                            name='transform_bent_NoisySRAM_31415_%i_%f' % (votes, sram_noise)
                        ),
                        'combiner': 'xor',
                        'bias': None,
                        'weight_random_seed': 31415,
                        'sigma_noise': arbiter_noise,
                    }
                )
            )
            for sram_noise in [0, .005, .01, .05, .1, .2, .5]
            for arbiter_noise in [0, 0.02, 0.1, .3, .5, .75, 1, 5, 10, 64]
            for k in [1]  #, 2, 4, 6, 8]
            for votes in [1, 3, 5, 11, 21]
        ]

    def plot(self):
        transform_name_pattern = r'transform_bent_NoisySRAM_[0-9]+_([0-9]+)_([0-9.]+).*'
        data = self.experimenter.results

        def from_transform_name(group, type_conversion=str):
            return data.apply(lambda row: type_conversion(re.sub(
                transform_name_pattern,
                r'\%i' % group,
                str(row['param_ins_gen__transformation'])
            )), axis=1)

        data['sram_noise'] = from_transform_name(group=2, type_conversion=float)
        data['votes'] = from_transform_name(group=1, type_conversion=int)
        data['arbiter_noise'] = data.apply(lambda row: row['param_ins_gen__sigma_noise'], axis=1)
        data['reliability_mean'] = data.apply(lambda row: 1 - row['mean'], axis=1)
        data['k'] = data.apply(lambda row: row['param_ins_gen__k'], axis=1)

        facet = catplot(
            x='sram_noise',
            y='reliability_mean',
            col='arbiter_noise',
            hue='votes',
            kind='bar',
            data=data,
        )

        #facet.set_axis_labels('SRAM Noise Level', 'Reliability')
        facet.fig.subplots_adjust(top=.8, wspace=.02, hspace=.02);
        facet.fig.suptitle('Bent XOR Arbiter PUF Reliabilty')
        facet.fig.savefig('figures/%s.pdf' % self.name(), bbox_inches='tight', pad_inches=.5)
