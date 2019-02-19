"""
This module provides an abstract class which defines an interface which an simulation have to implement in order to
interact with other classes in this project.
"""
import abc


class Simulation(object, metaclass=abc.ABCMeta):
    """
    This class is an interface which an simulation must implement in order to be compatible to other classes especially
    learner. This interface is designed to be used with challenge response simulations. A simulation gets some
    challenges as input in order to evaluate them. The return values of the evaluation are the responses of the PUF.
    """
    @abc.abstractmethod
    def eval(self, challenges):
        """
        This is the function which evaluates the PUF simulation.
        :param challenges: the type is not so strict and depends on the implementation
                       Input challenges
        :return the type is not so strict and depends on the implementation
                Response of the PUF simulation
        """
        raise NotImplementedError('users must define learn to use this base class')
