"""
Base module for studies.

pypuf studies are used to prove or support various claims. Each study demonstrates
a different result.
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
            cpu_limit=cpu_limit
        )

        # Expose results
        self.results = self.experimenter.results

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
        Generates this study's output
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
