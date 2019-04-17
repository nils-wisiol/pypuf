from pypuf.studies.base import Study
from pypuf import tools
from pypuf.learner.regression.logistic_regression import LogisticRegression
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from matplotlib.pyplot import figure
import seaborn as sns
from pypuf.experiments.experiment.base import Experiment
from numpy.random import RandomState
from os import getpid
from typing import NamedTuple
from uuid import UUID
from scipy.special import comb


class HybridAccuracyParameters(NamedTuple):
    """
    Experiment parameters for LTFArray.eval benchmark.
    """
    n: int
    k: int
    N: int
    transform: str
    combiner: str
    seed_instance: int


class Result(NamedTuple):
    """
    Experiment result from LTFArray.eval benchmark.
    """
    experiment_id: UUID
    pid: int
    accuracy: float


class HybridAccuracyExperiment(Experiment):
    """
    Measures the time the logistic regression learner takes to learn a target.
    """

    def __init__(self, progress_log_prefix, parameters):
        super().__init__(progress_log_prefix, parameters)
        self.instance = None
        self.learner = None
        self.model = None

    def prepare(self):
        # # change to not hardcoded
        # k = self.parameters.k  # 2
        # n = self.parameters.n  # 64
        # N = self.parameters.N  # 1200
        # transform = LTFArray.transform_id  # self.parameters.transform   #
        # combiner = self.parameters.combiner  # LTFArray.combiner_xor
        #
        # self.instance = LTFArray(
        #     weight_array=LTFArray.normal_weights(n=n, k=k),  # do not change, can be simulated by learner
        #     transform=transform,  # has to match learner, otherwise learner cannot match
        #     combiner=combiner,  # do not change
        # )
        #
        # self.learner = LogisticRegression(
        #     t_set=tools.TrainingSetHybrid(instance=self.instance, N=N),  # 6200
        #     n=comb(n, 2, exact=True),  # n choose k_original/k_new = 2
        #     k=k // 2,  # k divided by 2
        #     transformation=transform,
        #     combiner=combiner,
        #     # convergence_decimals=4,
        # )

        # change to not hardcoded
        self.instance = LTFArray(
            weight_array=LTFArray.normal_weights(n=64, k=2),  # do not change, can be simulated by learner
            transform=LTFArray.transform_id,  # has to match learner, otherwise learner cannot match
            combiner=LTFArray.combiner_xor,  # do not change
        )

        self.learner = LogisticRegression(
            t_set=tools.TrainingSetHybrid(instance=self.instance, N=1200),  # 6200
            n=2016,  # n choose k_original/k_new = 2
            k=1,  # k divided by 2
            transformation=LTFArray.transform_id,
            combiner=LTFArray.combiner_xor,
            # convergence_decimals=4,
        )

    def run(self):
        self.model = self.learner.learn()

    def analyze(self):
        """
        Analyzes the learned result.
        """
        assert self.model is not None
        accuracy = 1.0 - tools.approx_dist_hybrid(
            self.instance,
            self.model,
            min(10000, 2 ** self.parameters.n),
            random_instance=RandomState(self.parameters.seed_instance),
        )

        return Result(
            experiment_id=self.id,
            pid=getpid(),
            accuracy=accuracy,
        )


class HybridAccuracy(Study):
    SAMPLE_SIZE = 100
    TRANSFORMS = ["id"]

    def __init__(self):
        super().__init__()

    def experiments(self):
        experiments = []

        experiments.extend([
                HybridAccuracyExperiment(
                    progress_log_prefix=None,
                    parameters=HybridAccuracyParameters(
                        n=64,
                        k=2,
                        N=1200,
                        # N=10000,
                        transform='id',
                        combiner='xor',
                        seed_instance=314159 + i,
                    )
                )
                for i in range(self.SAMPLE_SIZE)
            ])

        return experiments

    def plot(self):
        # prepare data
        data = self.experimenter.results['accuracy']
        print(data)

        # plot
        fig = figure()
        ax = fig.add_subplot(1, 1, 1)
        sns.set()
        ax = sns.distplot(data)

        ax.set_xlabel('Accuracy')
        ax.set_ylabel('Count')
        fig.suptitle('pypuf Hybrid Attack Accuracy Results')
        fig.savefig('figures/hybrid_attack_accuracy', bbox_inches='tight', pad_inches=.5)