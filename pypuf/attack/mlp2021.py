import logging
from typing import List, Optional

import numpy as np
import tensorflow as tf
from numpy import ndarray

from .base import OfflineAttack
from ..io import ChallengeResponseSet
from ..simulation import Simulation

logger = logging.getLogger(__name__)


class MLPAttack2021(OfflineAttack):
    """
    Multilayer-Perceptron modeling attack for XOR Arbiter PUFs.

    Inspired by the works of Alkatheiri and Zhuang [AZ17]_ and Aseeri et al. [AZA18]_, introduced by
    Mursi et al. [MTZAA20]_ and Wisiol et al. [WMSZ21]_.
    """

    class Model(Simulation):

        def __init__(self, keras_model: tf.keras.Model, challenge_length: int) -> None:
            super().__init__()
            self.keras_model = keras_model
            self._challenge_length = challenge_length

        @property
        def challenge_length(self) -> int:
            return self._challenge_length

        @property
        def response_length(self) -> int:
            return 1

        @property
        def weights(self) -> List[ndarray]:
            return [l.get_weights() for l in self.keras_model.layers]

        def eval(self, challenges: ndarray) -> ndarray:
            features = np.cumprod(np.fliplr(challenges), axis=1, dtype=np.int8)
            return np.sign(self.keras_model.predict(features))

    class EarlyStopCallback(tf.keras.callbacks.Callback):

        def __init__(self, loss_threshold: float, patience: int) -> None:
            super().__init__()
            self.loss_threshold = loss_threshold
            self.patience = patience
            self.default_patience = patience
            self.previous_val_loss = 0.0

        def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
            if not logs:
                logs = {}

            # Stop the training when the validation accuracy reached the threshold accuracy
            if float(logs.get('val_loss')) < float(self.loss_threshold):
                logger.warning(f"Stopping early as validation loss below threshold of {self.loss_threshold} "
                               f"was reached.")
                self.model.stop_training = True

            # Stop the training when the validation acc is not enhancing
            if float(logs.get('val_loss')) > float(self.previous_val_loss):
                self.patience -= 1
                if not self.patience:
                    logger.warning(f'Stopping early as validation loss did not improve for {self.default_patience} '
                                   f'epochs (not necessarily continuously).')
                    self.model.stop_training = True
            self.previous_val_loss = logs.get('val_loss')

            # Stop the training when overfitting occurs
            if logs.get('accuracy') - logs.get('val_accuracy') > .15:
                logger.warning("Stopping early as overfitting of more than 15% in accuracy was detected.")
                self.model.stop_training = True

    def __init__(self, crps: ChallengeResponseSet, seed: int, net: List[int], epochs: int, lr: float, bs: int,
                 early_stop: float, patience: int = None, activation_hl: str = 'tanh') -> None:
        """
        Initialize the Multilayer Perceptron modeling attack, using the parameters given.

        Note that the complexity of the attack depends crucially on the parameters defined here. The attack by
        Aseeri et al. [AZA18]_ uses a network size of :math:`(2^k, 2^k, 2^k)` to model :math:`k`-XOR Arbiter PUFs and
        the ReLU activation function. An advancement of this attack [WMSZ21]_ uses :math:`(2^{k-1}, 2^k, 2^{k-1})` and
        the tanh activation function to model the same with far less required challenge-response data.

        :param crps: Challenge-response data observed from the PUF under attack. 99% of CRP data will be used as
            training data, 1% will be used as validation set.
        :type crps: :class:`pypuf.io.ChallengeResponseSet`
        :param seed: Random seed for model initialization. Success of the attack may depend on the seed, in particular
            when little challenge-response data is used.
        :type seed: ``int``
        :param net: Hidden-layer sizes for the multilayer perceptron. Note that the layers are all *dense*, i.e. fully
            connected.
        :type net: ``List[int]``
        :param epochs: Maximum number of epochs performed.
        :type epochs: ``int``
        :param lr: Learning rate of the Adam optimizer used for optimization.
        :type lr: ``float``
        :param bs: Number of training examples that are processed together. Larger block size benefits from higher
            confidence of gradient direction and better computational performance, smaller block size benefits from
            earlier feedback of the weight adoption on following training steps.
        :type bs: ``int``
        :param early_stop: Training will stop when validation loss is below this threshold.
        :type early_stop: ``float``
        :param patience: Training will stop when validation loss did not improve for the given number of epochs.
            Counter is not reset after validation improved in one epoch.
        :type patience: ``Optional[int]``
        :param activation_hl: Activation function used on the hidden layers.
        :type activation_hl: ``str``
        """
        super().__init__(crps)
        self.crps = crps
        self.net = net
        self.epochs = epochs
        self.lr = lr
        self.bs = bs
        self.seed = seed
        self.early_stop = early_stop
        self.patience = patience or epochs
        self.activation_hl = activation_hl
        self._history = None

    @staticmethod
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return tf.keras.losses.binary_crossentropy(.5 - .5 * y_true, .5 - .5 * y_pred)

    @staticmethod
    def accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return tf.keras.metrics.binary_accuracy(.5 - .5 * y_true, .5 - .5 * y_pred)

    @property
    def history(self) -> Optional[dict]:
        """
        After :meth:`fit` was called, returns a dictionary that contains information about the training process.
        The dictionary contains lists of length corresponding to the number of executed epochs:

        - ``loss`` the training loss,
        - ``val_loss`` the validation loss,
        - ``accuracy`` the training accuracy, and
        - ``val_accuracy`` the validation accuracy.
        """
        return self._history

    def fit(self) -> Model:
        """
        Using tensorflow, runs the attack as configured and returns the obtained model.

        .. note::
            Tensorflow will write to stdout.

        .. todo::
            Currently, a copy of the challenges is created to compute the features for learning. This essentially
            doubles memory consumption. If the challenges can be overwritten, this can be performed in-situ to reduce
            memory footprint of the attack.

        :return: Model of the XOR Arbiter PUF under attack.
        """
        tf.random.set_seed(self.seed)

        # prepare features and labels
        # TODO use pypuf's XORArbiterPUF.transform_atf to compute features
        # TODO consider in-situ computation of features
        features = np.cumprod(np.fliplr(self.crps.challenges), axis=1, dtype=np.int8)
        labels = self.crps.responses

        # build network
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(self.net[0], activation=self.activation_hl,
                                        input_dim=self.crps.challenge_length, kernel_initializer='random_normal'))
        for layer in self.net[1:]:
            model.add(tf.keras.layers.Dense(layer, activation=self.activation_hl))
        model.add(tf.keras.layers.Dense(1, activation='tanh'))
        opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
        model.compile(optimizer=opt, loss=self.loss, metrics=[self.accuracy])

        # fit
        self._history = model.fit(
            features, labels,
            epochs=self.epochs,
            batch_size=self.bs,
            callbacks=[self.EarlyStopCallback(self.early_stop, self.patience)],
            shuffle=True,
            validation_split=0.01,
        ).history

        # create pypuf model
        self._model = self.Model(model, challenge_length=self.crps.challenge_length)
        return self._model
