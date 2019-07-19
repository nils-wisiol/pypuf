"""
    This study runs the Perceptron Testing Experiment.
"""
from pypuf.experiments.experiment.test_perceptron import TestPerceptron
from pypuf.experiments.experiment.test_perceptron import Parameters

from pypuf.studies.base import Study


class PerceptronTesting(Study):
    """
        Perceptron testing study - GPU usage is possible.
    """
    def __init__(self):
        super().__init__(cpu_limit=1, gpu_limit=None)

        import tensorflow as tf
        tf.config.gpu.set_per_process_memory_growth(True)

    def experiments(self):
        experiments = []
        params = Parameters(n=32, k=2, N=1000,
                            batch_size=100,
                            epochs=100)
        e = TestPerceptron(
            progress_log_prefix=None,
            parameters=params
        )
        experiments.append(e)
        return experiments
