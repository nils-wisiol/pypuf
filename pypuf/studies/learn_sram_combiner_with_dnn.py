"""
    This study tries to learn a k-Arbiter PUF with SRAM Combiner using DNN learner.
"""
from pypuf.experiments.experiment.learn_sram_combiner_with_dnn import SRAMDNN
from pypuf.experiments.experiment.learn_sram_combiner_with_dnn import Parameters

from pypuf.studies.base import Study

#Plotting
from seaborn import catplot, axes_style


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
        for k, N in [(9,4e5)]:#,(5,8e5), (6,2e6)]:#,(7,7e6)]:
            params = Parameters(n=128, k=k, N=int(N),
                                batch_size=1000,
                                epochs=100)
            e = SRAMDNN(
                progress_log_prefix=None,
                parameters=params
            )
            experiments.append(e)
        return experiments

    def plot(self):
        data = self.experimenter.results

        with axes_style("whitegrid"):
            if not data.empty:
                facet = catplot(
                    x='k',
                    y='accuracy',
                    kind='point',
                    data=data
                )
                facet.set_axis_labels('Number of arbiter PUFs k', 'Neural Net Accuracy')
                facet.fig.set_size_inches(12, 4)
                facet.fig.subplots_adjust(top=.8, wspace=.02, hspace=.02)
                facet.fig.suptitle('Arbiter PUF with SRAM Combiner')
                facet.fig.savefig('figures/%s.sramcomb.pdf' % self.name(), bbox_inches='tight', pad_inches=.5)

