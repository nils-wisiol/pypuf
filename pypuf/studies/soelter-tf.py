from pypuf.experiments.experiment.soelter-tf import TestSoelter
from pypuf.experiments.experiment.soelter-tf import Parameters

from pypuf.studies.base import Study


class PerceptronTesting(Study):
    """
        Perceptron testing study - GPU usage is possible.
    """
    def __init__(self):
        super().__init__(cpu_limit=None, gpu_limit=2)

    def experiments(self):
        experiments = []
        params = Parameters(n=32, k=1, N=100000,
                            batch_size=100,
                            epochs=100)
        e = TestSoelter(
            progress_log_prefix=None,
            parameters=params
        )
        experiments.append(e)
        return experiments
