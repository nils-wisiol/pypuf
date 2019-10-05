from pypuf.experiments.experiment.reliability_based_cmaes import ExperimentReliabilityBasedCMAES, Parameters
from pypuf.studies.base import Study


class ReliabilityAttackStudy(Study):

    SHUFFLE = True

    def experiments(self):
        return [
            ExperimentReliabilityBasedCMAES(
                progress_log_name=None,
                parameters=Parameters(
                    n=n,
                    k=k,
                    seed_instance=seed,
                    seed_model=seed + 1,
                    seed_challenges=seed,
                    transform=transform,
                    combiner='xor',
                    noisiness=noisiness,
                    num=N,
                    reps=R,
                    pop_size=pop_size,
                    limit_stag=20,
                    limit_iter=500,
                )
            )
            for n in [64]
            for k in [1, 2, 3, 4, 5, 6]
            for transform in ['atf', 'id', 'lightweight_secure', 'fixed_permutation']
            for noisiness in [.01, .1, .25, .5]
            for N in [10**4, 10**5, 5 * 10**5, 10**6]
            for R in [3, 5, 11, 19, 49]
            for pop_size in [20, 50, 90]
            for seed in range(10)
        ]
