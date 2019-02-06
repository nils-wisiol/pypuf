"""
Accuracy Distribution of LR and Correlation Attacks on 4-XOR Arbiter PUFs

We conduct logistic regression and correlation attacks on XOR Arbiter PUFs
with classic, permutation, lightweight secure and pseudorandom input transformation,
with 500 samples each. We then output the distribution of resulting accuracies
shown by histograms.
"""

from matplotlib import pyplot
from numpy import arange, ones_like
from seaborn import distplot, set_context

from pypuf.experiments.experiment.correlation_attack import ExperimentCorrelationAttack, Parameters as CorrParameters
from pypuf.experiments.experiment.logistic_regression import ExperimentLogisticRegression, Parameters as LRParameters
from pypuf.studies.base import Study


class AccuracyDistributionStudy(Study):
    """
    Accuracy Distribution of LR and Correlation Attacks on 4-XOR Arbiter PUFs
    """

    CRPS = 30000
    SAMPLE_SIZE = 500
    LR_TRANSFORMATIONS = ['atf', 'random', 'lightweight_secure', 'fixed_permutation']
    CORR_TRANSFORMATIONS = ['lightweight_secure']
    SIZE = (64, 4)
    FIGURE_ORDER = {
        ('ExperimentLogisticRegression', 'lightweight_secure'): (1, 2, 1),
        ('ExperimentCorrelationAttack', None): (1, 2, 2),
    }

    SHUFFLE = True
    COMPRESSION = True

    NICE_TRANSFORMATION_NAMES = {
        'atf': 'Classic',
        'fixed_permutation': 'Permutation-Based',
        'lightweight_secure': 'Lightweight Secure',
        'random': 'Pseudorandom',
    }

    def experiments(self):
        e = []
        (n, k) = self.SIZE
        for transformation in self.LR_TRANSFORMATIONS:
            for i in range(self.SAMPLE_SIZE):
                e.append(
                    ExperimentLogisticRegression(
                        progress_log_prefix=None,
                        parameters=LRParameters(
                            seed_instance=314159 + i,
                            seed_model=265358 + i,
                            seed_challenge=979323 + i,
                            seed_distance=846264 + i,
                            n=n,
                            k=k,
                            transformation=transformation,
                            combiner='xor',
                            N=self.CRPS,
                            mini_batch_size=0,
                            convergence_decimals=2,
                            shuffle=False,
                        )
                    )
                )

        for i in range(self.SAMPLE_SIZE):
            e.append(
                ExperimentCorrelationAttack(
                    progress_log_prefix=None,
                    parameters=CorrParameters(
                        seed_instance=314159 + i,
                        seed_model=265358 + i,
                        seed_challenge=979323 + i,
                        seed_distance=846264 + i,
                        n=n,
                        k=k,
                        N=self.CRPS,
                        lr_iteration_limit=1000,
                        mini_batch_size=0,
                        convergence_decimals=2,
                        shuffle=False,
                    )
                )
            )
        return e

    def plot(self):
        data = self.experimenter.results
        set_context('paper', font_scale=.8)
        figure = pyplot.figure()
        figure.set_size_inches(w=6, h=1.75)
        figure.subplots_adjust(hspace=.1, wspace=.1)
        for (experiment, transformation), loc in self.FIGURE_ORDER.items():
            group_results = data[((data['transformation'] == transformation) | (transformation is None))
                                 & (data['experiment'] == experiment)]
            axis = figure.add_subplot(*loc)
            title = '{} using {:,} CRPs\n({} Attack)'.format(
                self.NICE_TRANSFORMATION_NAMES[transformation or 'lightweight_secure'],
                self.CRPS,
                'LR' if 'Correlation' not in experiment else 'Correlation'
            )
            axis.set_title(title)
            axis.set_xlim([.48, 1])
            distplot(
                group_results[['accuracy']],
                ax=axis,
                kde=False,
                bins=arange(.48, 1.01, .01),
                hist=True,
                norm_hist=False,
                color='blue',
                hist_kws={
                    'alpha': 1,
                    # the following line turns the histogram of absolute frequencies into one with relative frequencies
                    'weights': ones_like(group_results[['accuracy']]) / float(len(group_results[['accuracy']]))
                }
            )
        # for c in [0, 1]:
        #     axes[-1, c].set_xlabel('accuracy')
        # for r in range((len(subplots) + 1) // 2):
        #     axes[r, 0].set_ylabel('rel. frequency')
        figure.tight_layout()
        figure.savefig('figures/' + self.name() + '.fig_04.pdf')
        pyplot.close(figure)
