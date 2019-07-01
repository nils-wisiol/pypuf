"""
Simulations of a Physically Unclonable Functions (PUFs).
"""
import abc

from numpy import ndarray


class Simulation(object, metaclass=abc.ABCMeta):
    """
    Abstract base class for simulations of a Physically Unclonable Functions.
    """

    @abc.abstractmethod
    def challenge_length(self) -> int:
        """
        Returns the challenge length to used with the simulation. Note that the challenge length can be zero if
        the simulation does not accept any challenges.
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def response_length(self) -> int:
        """
        Returns the response length this simulation will provide per challenge.
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def eval(self, challenges: ndarray) -> ndarray:
        """
        Evaluate the PUF on a list of given challenges.
        :param challenges: List of challenges to evaluate on. Challenges must be given as ndarray of shape (N, n), where
        N is the number of challenges to be evaluated, and n must match Simulation.challenge_length(). Evaluating many
        challenges at once may have performance benefits, to evaluate a single challenge, provide an ndarray with shape
        (1, n). In cases where n = 0, an empty array with shape (N, 0) needs to be provided to determine the number of
        responses requested.
        :return ndarray of shape (N, m), listing the simulated responses to the challenges in order they were given,
        where m must match Simulation.response_length.
        """
        raise NotImplementedError()
