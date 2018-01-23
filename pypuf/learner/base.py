"""
This module provides an abstract class which can be used to implement a learning algorithm which is compatible to
other modules of the pypuf project. The scope of this module are algorithms which model a PUF instance with a given
trainingset of challenges and responses.
"""
import abc


class Learner(object, metaclass=abc.ABCMeta):
    """
    This class is the base class for all learning classes.
    """
    @abc.abstractmethod
    def learn(self):
        """
        This function is the core of the learning class. This function should implement all necessary step which are
        need to learn a PUF instance.
        return: pypuf.simulation.base.Simulation
                This is an instance of a simulation class which was learned by the learner implementation.
        """
        raise NotImplementedError('users must define learn to use this base class')
