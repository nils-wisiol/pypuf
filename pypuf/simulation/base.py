import abc


class Simulation(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def eval(self):
        raise NotImplementedError('users must define learn to use this base class')