import os

from pypuf.studies.base import Study
from pypuf.experiments.experiment.custom_reliability_based_ipuf_layer_attack import \
    ExperimentCustomReliabilityBasedLayerIPUF, Parameters


class CustomLayerIPUFAttackStudy(Study):

    def __init__(self):
        super().__init__(gpu_limit=2)
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    def experiments(self):
        return [
            ExperimentCustomReliabilityBasedLayerIPUF(
                progress_log_name=None,
                parameters=Parameters(
                    n=n,
                    k_up=k_up,
                    k_down=k_down,
                    layer=layer,
                    seed=seed,
                    noisiness=noisiness,
                    N=N,
                    R=R,
                    eps=eps,
                    extra=extra,
                    remove_error_1=remove_error_1,
                    remove_error_2=remove_error_2,
                    abort_delta=1e-4,
                    max_tries=1,
                    gpu_id=1,
                    separate=separate,
                    heuristic=heuristic,
                ),
            )
            for n in [64]
            for noisiness in [0.1]
            for k_up, k_down, extra, N in [
                # (1, 2, 10, 150000),
                (1, 3, 10, 200000),
                (1, 4, 10, 400000),
                # (1, 5, 10, 1000000),
                # (1, 6, 10, 3000000),
            ]
            for layer, heuristic in [
                ('upper', [1, 0, 1, 1]),    # best
                ('upper', [1, 0, 1, 2]),
                ('upper', [1, 0, 2, 1]),
                ('upper', [2, 0, 1, 1]),
                ('upper', [2, 0, 1, 2]),
                ('upper', [2, 0, 2, 1]),
                ('upper', [0, 2, 1, 1]),
                ('upper', [0, 2, 1, 2]),
                ('upper', [0, 2, 2, 1]),
            ]
            for remove_error_1, remove_error_2 in [
                (False, False),
            ]
            for R in [51]
            for eps in [0.9]
            for separate in [False]
            for seed in range(20)
        ]

    def plot(self):
        pass
