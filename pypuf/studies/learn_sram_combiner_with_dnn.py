"""
    This study tries to learn a k-Arbiter PUF with SRAM Combiner using DNN learner.
"""
from pypuf.experiments.experiment.learn_sram_combiner_with_dnn import SRAMDNN
from pypuf.experiments.experiment.learn_sram_combiner_with_dnn import Parameters

from pypuf.studies.base import Study


class SRAMCombinerLearning(Study):
    """
        SRAM Combiner learning study - GPU usage is possible.
    """
    def __init__(self):
        super().__init__(cpu_limit=1, gpu_limit=None)

        import tensorflow as tf
        tf.config.gpu.set_per_process_memory_growth(True)

    def experiments(self):
        experiments = []
        for k in [4,8,16]:
            params = Parameters(n=32, k=k, N=1000,
                                batch_size=100,
                                epochs=100)
            e = SRAMDNN(
                progress_log_prefix=None,
                parameters=params
            )
            experiments.append(e)
        return experiments
