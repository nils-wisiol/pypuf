import abc


class Learner(object, metaclass=abc.ABCMeta):
    @property
    def instance(self):
        raise NotImplementedError('users must define instance to use this base class')

    @instance.setter
    @abc.abstractmethod
    def instance(self, val):
        raise NotImplementedError('users must define instance to use this base class')

    @abc.abstractmethod
    def learn(self):
        raise NotImplementedError('users must define learn to use this base class')