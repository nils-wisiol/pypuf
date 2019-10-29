from os import getpid
from typing import NamedTuple
from uuid import UUID

import numpy
from keras import Sequential, Model
from keras.backend import mean, sign
from keras.layers import Dense, Activation
from matplotlib.pyplot import close
from numpy import around, average, zeros
import random
from seaborn import catplot, axes_style

from pypuf.experiments.experiment.base import Experiment
from pypuf.simulation.arbiter_based.arbiter_puf import XORArbiterPUF
from pypuf.studies.base import Study
from pypuf.tools import TrainingSet, ChallengeResponseSet, approx_dist_nonrandom


class Parameters(NamedTuple):
    N: int
    n: int
    k: int
    transform: str
    seed: int


class Result(NamedTuple):
    experiment_id: UUID
    measured_time: int
    pid: int
    accuracy: float
    max_memory: int


class TFExperiment(Experiment):

    training_set: ChallengeResponseSet
    test_set: ChallengeResponseSet
    model: Model

    def prepare(self):
        simulation = XORArbiterPUF(
            n=self.parameters.n,
            k=self.parameters.k,
            transform=self.parameters.transform,
            seed=self.parameters.seed,
        )
        self.training_set = TrainingSet(simulation, self.parameters.N)
        self.test_set = TrainingSet(simulation, 10**4)
        #for crp_set in [self.training_set, self.test_set]:
        #    crp_set.responses = .5 - .5 * crp_set.responses
        #    crp_set.challenges = .5 - .5 * crp_set.challenges

    @staticmethod
    def _accuracy(y_true, y_pred):
        return 1 - mean(y_true * sign(y_pred))

    def run(self):
        self.model = model = Sequential()
        model.add(Dense(1, input_dim=self.parameters.n))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='tanh'))
        model.summary()

        model.compile(loss='squared_hinge', optimizer='adam', metrics=['accuracy', self._accuracy])

        model.fit(
            self.training_set.challenges, self.training_set.responses,
            epochs=15,
            batch_size=1000,
            validation_data=(self.test_set.challenges, self.test_set.responses),
        )

    def analyze(self):
        predicted = self.model.predict(self.test_set.challenges)
        print(predicted)
        #accuracy = (1 + numpy.average(numpy.sign(predicted * self.test_set.responses))) / 2
        accuracy = numpy.average(((numpy.sign(predicted) * self.test_set.responses) + 1) / 2)

        print(accuracy)
        comp = zeros(shape=(4,18))
        the_set = [random.randint(0, 10**4) for _ in range(18)]
        comp[0, :] = sign(predicted)[the_set, 0]
        comp[1, :] = self.test_set.responses[the_set]
        comp[2, :] = comp[0, :] * comp[1, :]
        comp[3, :] = (comp[2, :] + 1) / 2
        print(numpy.average(comp[3, :]))
        print(comp)

        return Result(
            experiment_id=self.id,
            measured_time=self.measured_time,
            pid=getpid(),
            accuracy=accuracy,
            max_memory=self.max_memory(),
        )


class InterposeMLPStudy(Study):

    EXPERIMENTER_CALLBACK_MIN_PAUSE = 10

    def experiments(self):
        return [
            TFExperiment(
                progress_log_name='tfe',
                parameters=Parameters(
                    N=N, n=n, k=k, transform='id', seed=seed,
                )
            )
            for n in [64]
            for k in [2]
            for N in [400000]
            for seed in range(1)
        ]

    def plot(self):
        data = self.experimenter.results
        data['max_memory_gb'] = data.apply(lambda row: row['max_memory'] / 1024**3, axis=1)
        data['Ne6'] = data.apply(lambda row: row['N'] / 1e6, axis=1)
        f = catplot(
            data=data,
            x='Ne6',
            y='accuracy',
            row='k',
            kind='swarm',
            aspect=7.5,
            height=1.2,
        )
        f.savefig(f'figures/{self.name()}.pdf')
        close(f.fig)
