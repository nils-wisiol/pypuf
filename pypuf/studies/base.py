"""
Base module for studies.

pypuf studies can be used to run experiments and analyze the results. Ideally, a study can answer a specific
research question like "How many examples does a training set need to have?" or "What's the success rate of
an attack?".

To run experiments, the pypuf `Experimenter` is used. Results and partial results will be saved to disk.
While and after experiments are run, studies can analyze the results by plotting graphs. This enables the user
to rapidly obtain results that will become more and more exact over time.

In order to implement a study, the `name`, `experiments` and `plot` method should be implemented.

- The `name` method returns a single, constant string that serves as the studies name and is used in file names to
  identify the study.
- The `experiments` method returns an array of Experiments that will be run by the experimenter.
- The `plot` method has access to the (full or partial) results in a `DataFrame` via `self.experimenter.results` and
  analyzes and plots the results.
"""
from pypuf.experiments import Experimenter


class Study:
    """
    Creates a number of experiments, runs them and produces a human-readable result.
    """

    EXPERIMENTER_CALLBACK_MIN_PAUSE = 5 * 60
    SHUFFLE = False
    COMPRESSION = False
    STUDY_MODULE_PREFIX = 'pypuf.studies.'

    def __init__(self, cpu_limit=None, gpu_limit=None):
        """
        Initialize the study.
        """
        self.results = None
        self.full_run = True

        # Callback method
        def callback(experiment_id):
            self._callback(experiment_id)

        # Create experimenter
        self.experimenter = Experimenter(
            self.name(),
            update_callback=callback,
            update_callback_min_pause=self.EXPERIMENTER_CALLBACK_MIN_PAUSE,
            cpu_limit=cpu_limit,
            gpu_limit=gpu_limit,
            results_file=self.name() + ('.csv.gz' if self.COMPRESSION else '.csv'),
        )

    def name(self):
        """
        returns the study's name
        """
        name = self.__module__.__str__()
        if name.startswith(self.STUDY_MODULE_PREFIX):
            name = name[len(self.STUDY_MODULE_PREFIX):]
        return name

    def experiments(self):
        """
        returns the study's experiments as a list
        """
        return []

    def plot(self):
        """
        Generates this study's output. (Full or partial) results are available via `self.experimenter.results`.
        """

    def run(self, part=0, total=1):
        """
        runs the study, that is, create the experiments, run them through an experimenter,
        collect the results and plot the output
        """
        # Queue experiments
        experiments = self.experiments()
        assert experiments, 'Study {} did not define any experiments.'.format(self.name())
        partition_size = len(experiments) // total
        self.full_run = partition_size == len(experiments)
        start = part * partition_size
        end = (part + 1) * partition_size
        for e in experiments[start:end]:
            self.experimenter.queue(e)

        # Run experiments
        if experiments[start:end]:
            self.experimenter.run(shuffle=self.SHUFFLE)

            # Plot results
            self.plot()
        else:
            print(f'Part {part} of {total} parts did not have any experiments, study defines a total of '
                  f'{len(experiments)} experiments.')

    def _callback(self, _experiment_id=None):
        """
        will be called by the experimenter after every finished experiment,
        but at most every EXPERIMENTER_CALLBACK_MIN_PAUSE seconds.
        """
        if not self.experimenter.results.empty and self.full_run:
            self.plot()
