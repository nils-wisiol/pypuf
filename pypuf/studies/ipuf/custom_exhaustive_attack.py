import os

from pypuf.studies.base import Study
from pypuf.experiments.experiment.custom_reliability_based_ipuf_layer_attack import \
    ExperimentCustomReliabilityBasedLayerIPUF, Parameters


class CustomLayerIPUFAttackStudy(Study):

    def __init__(self):
        super().__init__(gpu_limit=2)
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    def experiments(self):
        k = 10 ** 3
        M = 10 ** 6
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
                    fitness=fitness,
                ),
            )
            for noisiness in [0.1]
            # max number of chains that are learned is k_up + extra
            for n, k_up, k_down, extra, N in [
                # (64, 2, 2, 2, 200 * k),
                # (64, 4, 4, 12, 2 * M),
                # (64, 2, 8, 8, 2 * M),
                # (64, 8, 1, 8, 2 * M),
                (64, 1, 8, 11, 2 * M),      # 10h   0.5GB
                (64, 8, 8, 4, 4 * M),       # 10h   0.5GB
                (64, 8, 1, 4, 2 * M),       # 10h   0.5GB
                # (64, 1, 8, 7, 500 * k),

                # (64, 1, 16, 7, 5 * M),      # 5h    0.5GB
                # (64, 2, 16, 10, 10 * M),    # 10h   1.25GB
                # (64, 4, 16, 12, 20 * M),    # 40h   2.5GB
                # (64, 8, 16, 16, 40 * M),    # 160h  5GB
                # (64, 1, 32, 7, 10 * M),     # 10h   1.25GB
                # (64, 2, 32, 10, 20 * M),    # 20h   2.5GB
                # (64, 4, 32, 12, 40 * M),    # 80h   5GB

                # (128, 1, 16, 7, 10 * M),    # 20h   1.25GB
                # (128, 2, 16, 10, 20 * M),   # 40h   2.5GB
                # (128, 4, 16, 12, 40 * M),   # 160h  5GB
                # (128, 1, 32, 7, 20 * M),    # 20h   2.5GB
                # (128, 2, 32, 10, 40 * M),   # 40h   5GB
            ]
            for fitness in [
                'penalty',
                # 'combine',
                # 'remove',
            ]
            for layer, heuristic in [
                ('upper', [1, 0, 1, 1]),
                ('upper', [1, 0, 1, 2]),
                ('upper', [1, 0, 2, 1]),
                ('upper', [2, 0, 1, 1]),
                ('upper', [2, 0, 1, 2]),
                ('upper', [2, 0, 2, 1]),
                ('upper', [0, 1, 1, 1]),
                ('upper', [0, 1, 1, 2]),
                ('upper', [0, 1, 2, 1]),
                ('upper', [0, 2, 1, 1]),
                ('upper', [0, 2, 1, 2]),
                ('upper', [0, 2, 2, 1]),
                ('upper', [1, 1, 1, 0]),
                ('upper', [1, 2, 1, 0]),
                ('upper', [2, 1, 1, 0]),
                ('upper', [1, 1, 2, 0]),
                ('upper', [1, 2, 2, 0]),
                ('upper', [2, 1, 2, 0]),
                ('upper', [1, 1, 0, 1]),
                ('upper', [1, 2, 0, 1]),
                ('upper', [2, 1, 0, 1]),
                ('upper', [1, 1, 0, 2]),
                ('upper', [1, 2, 0, 2]),
                ('upper', [2, 1, 0, 2]),
            ]
            for remove_error_1, remove_error_2 in [
                (False, False),
            ]
            for R in [51]
            for eps in [0.9]
            for separate in [False]
            for seed in range(10)
        ]

    def plot(self):
        pass
