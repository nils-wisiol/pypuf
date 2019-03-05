from pypuf.experiments.experiment.correlation_attack import ExperimentCorrelationAttack, Parameters as CorrParameters
from pypuf.experiments.experiment.logistic_regression import ExperimentLogisticRegression, Parameters as LRParameters
from pypuf.studies.base import Study
from matplotlib import pyplot
from seaborn import distplot
from numpy import arange, ones_like


class Fig04(Study):

    CRPS = 30000
    SAMPLE_SIZE = 500
    LR_TRANSFORMATIONS = ['atf', 'random', 'lightweight_secure', 'fixed_permutation']
    CORR_TRANSFORMATIONS = ['lightweight_secure']
    SIZE = (64, 4)
    FIGURE_ORDER = {
        ('ExperimentLogisticRegression', 'atf'): 0,
        ('ExperimentLogisticRegression', 'random'): 1,
        ('ExperimentLogisticRegression', 'lightweight_secure'): 2,
        ('ExperimentCorrelationAttack', 'lightweight_secure'): 3,
        ('ExperimentLogisticRegression', 'fixed_permutation'): 4,
    }

    SHUFFLE = True

    NICE_TRANSFORMATION_NAMES = {
        'atf': 'Classic',
        'fixed_permutation': 'Permutation-Based',
        'lightweight_secure': 'Lightweight Secure',
        'random': 'Pseudorandom',
    }

    def name(self):
        return 'fig_04'

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
        subplots = []
        experiment_groups = self.experimenter.results.groupby(['experiment'])
        for experiment, experiment_group in experiment_groups:
            if experiment == 'ExperimentLogisticRegression':
                transformation_groups = experiment_group.groupby(['transformation'])
                for transformation, transformation_group in transformation_groups:
                    subplots.append((experiment, transformation, transformation_group))
            else:
                subplots.append((experiment, 'lightweight_secure', experiment_group))

        subplots.sort(key=lambda x: self.FIGURE_ORDER[(x[0], x[1])])
        figure, axes = pyplot.subplots(nrows=len(subplots), ncols=1)
        figure.subplots_adjust(hspace=3)
        figure.set_size_inches(w=5, h=1.5 * len(subplots))
        for axis, (experiment, transformation, group_results) in zip(axes, subplots):
            title = '{} using {:,} CRPs'.format(self.NICE_TRANSFORMATION_NAMES[transformation], self.CRPS)
            if experiment == 'ExperimentCorrelationAttack':
                title += ' (Improved Attack)'
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
        axes[-1].set_xlabel('accuracy')
        axes[(len(subplots) - 1) // 2].set_ylabel('rel. frequency')
        figure.tight_layout()
        figure.savefig('figures/' + self.name() + '.pdf')
        pyplot.close(figure)
