from typing import Optional, List

import numpy as np
import tensorflow as tf

from .base import OfflineAttack
from ..io import ChallengeResponseSet
from ..simulation import XORArbiterPUF, BeliPUF, TwoBitBeliPUF, OneBitBeliPUF
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
                 stop_validation_accuracy: float = .95, validation_set_size: float = .01) -> None:
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
        :param validation_set_size: Proportion of CRPs to be used for validation, if <= 1, or else absolute number of
            CRPs used to validation.
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
        self._keras_model = None
        self.validation_split = validation_set_size if validation_set_size <= 1 else validation_set_size / len(crps)

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

    def fit(self, verbose: bool = True) -> Simulation:
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

        self._keras_model = model = tf.keras.Model(inputs=[input_tensor], outputs=[output])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss=self.loss,
            metrics=[self.accuracy],
        )

        self._history = model.fit(
            features, labels,
            batch_size=self.bs,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=[self.AccuracyStop(self.stop_validation_accuracy)],
            verbose=verbose,
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


class BeliLR(LRAttack2021):

    model_class = None

    @staticmethod
    def beli_output(output_delays: List[tf.Tensor]) -> List[tf.Tensor]:
        raise NotImplementedError

    def beli_model(self, input_tensor: tf.Tensor) -> List[tf.Tensor]:
        internal_delays = tf.keras.layers.Dense(
            units=1,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=10, stddev=.05),
            activation=None,
        )
        output_delays = [internal_delays(input_tensor[:, i]) for i in range(input_tensor.shape[1])]
        return self.beli_output(output_delays)

    def fit(self, verbose: bool = True) -> Simulation:
        """
        Using tensorflow, runs the attack as configured and returns the obtained model.

        :param verbose: If true (the default), tensorflow will write progress information to stdout.
        :return: Model of the Beli PUF under attack.
        """
        tf.random.set_seed(self.seed)

        n = self.crps.challenge_length
        k = self.k

        input_tensor = tf.keras.Input(shape=(4, 4 * n))
        beli_models = tf.transpose(
            [self.beli_model(input_tensor) for _ in range(k)],  # by list comprehension, k-axis is axis 0
            (2, 1, 0, 3)  # swap k-axis to axis 2 and keep sample axis at axis 0
        )  # output dim: (sample, m, k, 1)
        xor = tf.math.reduce_prod(beli_models, axis=2)  # xor along k-axis
        outputs = tf.keras.layers.Activation(tf.keras.activations.tanh)(xor)

        model = tf.keras.Model(inputs=input_tensor, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adadelta(learning_rate=self.lr),
            loss=self.loss,
            metrics=[self.accuracy],
        )
        self._keras_model = model

        self._history = model.fit(
            BeliPUF.features(self.crps.challenges),
            self.crps.responses,
            batch_size=self.bs,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=[self.AccuracyStop(self.stop_validation_accuracy)],
            verbose=verbose,
        ).history

        self._model = self.keras_to_pypuf(model)
        return self.model

    def keras_to_pypuf(self, keras_model: tf.keras.Model) -> LTFArray:
        """
        Given a Keras model that resulted from the attack of the :meth:`fit` method, constructs an
        :class:`pypuf.simulation.BeliPUF` that computes the same model.
        """
        delays = np.array([
            layer.get_weights()[0].squeeze().reshape((8, -1))
            for layer in keras_model.layers
            if isinstance(layer, tf.keras.layers.Dense)]
        )

        k = delays.shape[0]
        n = delays.shape[2] * 2

        pypuf_model = self.model_class(n=n, k=k, seed=0)
        pypuf_model.delays = delays

        return pypuf_model


class TwoBitBeliLR(BeliLR):
    model_class = TwoBitBeliPUF

    @staticmethod
    def beli_output(output_delays: List[tf.Tensor]) -> List[tf.Tensor]:
        r"""
        Returns a continuous estimate of the two output bits.

        Let :math:`d_i` be the delay on delay line :math:`i`.

        For the first bit, :math:`\min\{d_2,d_3\} - \min\{d_0,d_1\}` is returned.
        This expression is positive if and only if :math:`d_0` or :math:`d_1` is the fastest signal.
        As first response bit is positive if delay :math:`d_0` or :math:`d_1` is fastest,
        this expression is an approximation of the first response bit value.

        A similar argument holds for the second response bit.
        """
        Min = tf.keras.layers.Minimum
        d = output_delays
        return [
            Min()((d[2], d[3])) - Min()((d[0], d[1])),
            Min()((d[1], d[3])) - Min()((d[0], d[2])),
        ]


class OneBitBeliLR(BeliLR):
    model_class = OneBitBeliPUF

    @staticmethod
    def beli_output(output_delays: List[tf.Tensor]) -> List[tf.Tensor]:
        r"""
        Returns a continuous estimate of the output bit.

        Let :math:`d_i` be the delay on delay line :math:`i`.

        :math:`\min\{d_1,d_2\} - \min\{d_0,d_3\}` is returned.
        This expression is positive if and only if :math:`d_0` or :math:`d_3` is the fastest signal.
        As the response bit is positive if and only if the delay :math:`d_0` or :math:`d_3` is fastest,
        this expression is an approximation of the response bit value.
        """
        Min = tf.keras.layers.Minimum
        d = output_delays
        return [
            Min()((d[1], d[2])) - Min()((d[0], d[3])),
        ]
