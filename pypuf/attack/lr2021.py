from typing import Optional

import numpy as np
import tensorflow as tf

from .base import OfflineAttack
from ..io import ChallengeResponseSet
from ..simulation import XORArbiterPUF
from ..simulation.base import Simulation, LTFArray


class LRAttack2021(OfflineAttack):
    """
    Improved Logistic Regression modeling attack for XOR Arbiter PUFs.

    Based on the attack of Rührmair et al. [RSSD10]_, this version uses tensorflow to model XOR Arbiter PUFs based on
    observed challenge-response pairs. Compared to the version used by the original authors, this version is based on
    tensorflow and uses some detail improvements.

    .. todo::
        A detailed description of the modifications used in pypuf is currently under consideration for publication.
        This section will be updated as soon as the manuscript is available to the public.
    """

    class AccuracyStop(tf.keras.callbacks.Callback):

        def __init__(self, stop_validation_accuracy: float) -> None:
            super().__init__()
            self.stop_validation_accuracy = stop_validation_accuracy

        def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
            if float(logs.get('val_accuracy')) > self.stop_validation_accuracy:
                self.model.stop_training = True

    def __init__(self, crps: ChallengeResponseSet, seed: int, k: int, bs: int, lr: float, epochs: int,
                 stop_validation_accuracy: float = .95) -> None:
        """
        Initialize an improved Logistic Regression attack using the given parameters.

        :param crps: Challenge-response data observed from the PUF under attack. 99% of CRP data will be used as
            training data, 1% will be used as validation set.
        :type crps: :class:`pypuf.io.ChallengeResponseSet`
        :param seed: Random seed for model initialization. Success of the attack may depend on the seed, in particular
            when little challenge-response data is used.
        :type seed: ``int``
        :param k: Number of parallel arbiter chains used in the XOR Arbiter PUF.
        :type k: ``int``
        :param bs: Number of training examples that are processed together. Larger block size benefits from higher
            confidence of gradient direction and better computational performance, smaller block size benefits from
            earlier feedback of the weight adoption on following training steps.
        :type bs: ``int``
        :param lr: Learning rate of the Adam optimizer used for optimization.
        :type lr: ``float``
        :param epochs: Maximum number of epochs performed.
        :type epochs: ``int``
        :param stop_validation_accuracy: Training is stopped when this validation accuracy is reached. Set to 1 to
            deactivate.
        :type stop_validation_accuracy: ``float``
        """
        super().__init__(crps)
        self.crps = crps
        self.seed = seed
        self.k = k
        self.bs = bs
        self.lr = lr
        self.epochs = epochs
        self.stop_validation_accuracy = stop_validation_accuracy
        self._history = None

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

    @staticmethod
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return tf.keras.losses.binary_crossentropy(.5 - .5 * y_true, .5 - .5 * y_pred)

    @staticmethod
    def accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return tf.keras.metrics.binary_accuracy(.5 - .5 * y_true, .5 - .5 * y_pred)

    def fit(self) -> Simulation:
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
        n, k = self.crps.challenges.shape[1], self.k

        features = XORArbiterPUF.transform_atf(self.crps.challenges, k=1)[:, 0, :]
        labels = self.crps.responses

        # TODO include in-situ option to save memory:
        # for i in range(n - 2, -1, -1):
        #     crps.challenges[:, i] *= crps.challenges[:, i + 1]
        # crps.responses[crps.responses == 1] = 0
        # crps.responses[crps.responses == -1] = 1

        input_tensor = tf.keras.Input(shape=(n,))
        if self.k > 1:
            prod = tf.keras.layers.Multiply()(
                [
                    tf.keras.layers.Dense(
                        units=1,
                        kernel_initializer=tf.keras.initializers.RandomNormal(),
                        bias_initializer=tf.keras.initializers.Zeros(),
                        activation=tf.keras.activations.tanh,
                    )(input_tensor) for _ in range(k)
                ]
            )
            output = tf.keras.layers.Activation('tanh')(prod)
        else:
            output = tf.keras.layers.Dense(
                units=1,
                kernel_initializer=tf.keras.initializers.RandomNormal(),
                bias_initializer=tf.keras.initializers.Zeros(),
                activation=tf.keras.activations.tanh,
            )(input_tensor)

        model = tf.keras.Model(inputs=[input_tensor], outputs=[output])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss=self.loss,
            metrics=[self.accuracy],
        )

        self._history = model.fit(
            features, labels,
            batch_size=self.bs,
            epochs=self.epochs,
            validation_split=.01,
            callbacks=[self.AccuracyStop(self.stop_validation_accuracy)],
        ).history

        self._model = self.keras_to_pypuf(model)

        return self.model

    def keras_to_pypuf(self, keras_model: tf.keras.Model) -> LTFArray:
        """
        Given a Keras model that resulted from the attack of the :meth:`fit` method, constructs an
        :class:`pypuf.simulation.LTFArray` that computes the same model.
        """
        n, k = self.crps.challenges.shape[1], self.k

        weights = np.zeros(shape=(k, n))
        bias = np.zeros(shape=(k,))

        for l in range(k):
            layer_weights = keras_model.layers[l + 1].get_weights()
            weights[l] = layer_weights[0][:, 0]
            bias[l] = layer_weights[1]

        return LTFArray(weight_array=weights, bias=bias, transform=XORArbiterPUF.transform_atf)
