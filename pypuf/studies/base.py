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

    EXPERIMENTER_CALLBACK_MIN_PAUSE = 30
    SHUFFLE = False

    def __init__(self, cpu_limit=None):
        """
        Initialize the study.
        """
        self.results = None

        # Callback method
        def callback(experiment_id):
            self._callback(experiment_id)

        # Create experimenter
        self.experimenter = Experimenter(
            self.name(),
            update_callback=callback,
            update_callback_min_pause=self.EXPERIMENTER_CALLBACK_MIN_PAUSE,
            cpu_limit=cpu_limit,
            results_file=self.name() + '_results.csv',
        )

    def name(self):
        """
        returns the study's name
        """

    def experiments(self):
        """
        returns the study's experiments as a list
        """
        return []

    def plot(self):
        """
        Generates this study's output. (Full or partial) results are available via `self.experimenter.results`.
        """

    def run(self):
        """
        runs the study, that is, create the experiments, run them through an experimenter,
        collect the results and plot the output
        """
        # Queue experiments
        for e in self.experiments():
            self.experimenter.queue(e)

        # Run experiments
        self.experimenter.run(shuffle=self.SHUFFLE)

        # Plot results
        self.plot()

    def _callback(self, _experiment_id=None):
        """
        will be called by the experimenter after every finished experiment,
        but at most every EXPERIMENTER_CALLBACK_MIN_PAUSE seconds.
        """
        self.plot()
