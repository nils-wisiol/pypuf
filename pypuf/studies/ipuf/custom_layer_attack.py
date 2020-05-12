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
                    max_tries=k_down,
                    gpu_id=1,
                )
            )
            for n in [64]
            for noisiness in [0.1]
            for k_up, k_down, extra, N in [
                # (1, 4, 2, 10000),
                # (4, 4, 2, 10000),
                # (1, 3, 2, 100000),
                (3, 3, 2, 400000),
                # (1, 4, 2, 1000000),
                # (4, 4, 2, 1000000),
            ]
            for layer in ['lower', 'upper']
            for remove_error_1, remove_error_2 in [
                (False, False),
                (False, True),
                (True, False),
                (True, True)
            ]
            for R in [51]
            for eps in [0.9]
            for seed in range(10)
        ]

    def plot(self):
        pass
