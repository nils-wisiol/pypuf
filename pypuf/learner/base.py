import abc


class Learner(object, metaclass=abc.ABCMeta):
    @property
    def training_set(self):
        raise NotImplementedError('users must define training_set to use this base class')

    @training_set.setter
    @abc.abstractmethod
    def training_set(self, val):
        raise NotImplementedError('users must define training_set to use this base class')

    @abc.abstractmethod
    def learn(self):
        raise NotImplementedError('users must define learn to use this base class')