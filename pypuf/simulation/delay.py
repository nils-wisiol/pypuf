from typing import Tuple
from typing import Union, Callable, List

import numpy as np
from numpy import sign, sum as np_sum, concatenate, ndarray, transpose, broadcast_to, swapaxes, array, copy
from numpy.random import default_rng, RandomState

from .base import Simulation, NoisyLTFArray, LTFArray, XORPUF


class SimulationMajorityLTFArray(NoisyLTFArray):
    """
    This class provides a majority vote version of the NoisyLTFArray.
    It uses different noises for each PUF instance and each challenge input.
    Majority vote means that each fo the k PUFs get evaluated vote_count times
    in order to mitigate the impact of noise to the responses. With this class
    it is possible to simulate quite stable huge PUF systems.
    This class can be used as PUF simulation in order to generate a trainingset.
    """

    def __init__(self, weight_array: ndarray, transform: Union[Callable, str], combiner: Union[Callable, str],
                 sigma_noise: float, seed: int, vote_count: int = 1, bias: ndarray = None) -> None:
        """
        :param weight_array: array of floats with shape(k,n)
                            Array of weights which represents the PUF stage delays.
        :param transform: A function: array of int with shape(N,n), int number of PUFs k -> shape(N,k,n)
                          The function transforms input challenges in order to increase resistance against attacks.
        :param combiner: A function: array of int with shape(N,k,n) -> array of in with shape(N)
                         The functions combines the outputs of k PUFs to one bit results,
                         in oder to increase resistance against attacks.
        :param sigma_noise: float
                            Standard deviation of noise distribution.
        :param seed: int
                      This pseudo-random number generator is used to generate noise.
        :param bias: None, float or a two dimensional array of float with shape (k, 1)
                     This bias value or array of bias values will be appended to the weight_array.
                     Use a single value if you want the same bias for all weight_vectors.
        :param vote_count: positive odd int
                           Number which defines the number of evaluations of PUFs in oder to majority vote the output.
        """
        assert vote_count % 2 == 1  # majority vote only works with an odd number of votes
        self.vote_count = vote_count
        super().__init__(weight_array, transform, combiner, sigma_noise, seed, bias=bias)

    def eval(self, challenges: ndarray, block_size: int = None) -> ndarray:
        # TODO Add block-wise eval support. It seems to work, but it changes the noise, as the noise PRNG will be called
        #  differently.
        assert block_size is None, f'{self.__class__.__name__} currently does not support block-wise evaluation.'
        return super().eval(challenges, block_size)

    def val(self, challenges: ndarray) -> ndarray:
        """
        This function a calculates the output of the LTFArray based on weights with majority vote.
        :param challenges: array of int shape(N,k,n)
                       Array of challenges which should be evaluated by the simulation.
        :return: array of int shape(N)
                 Array of responses for the N different challenges.
        """
        return self.combiner(self.majority_vote(self.transform(challenges, self.k)))

    def majority_vote(self, sub_challenges: ndarray) -> ndarray:
        """
        This function evaluates transformed input challenges and uses majority vote on them.
        :param sub_challenges: array of int with shape(N,k,n)
                                   Array of transformed input challenges.
        :return: array of int with shape(N,k,n)
                 Majority voted responses for each of the k PUFs.
        """
        # Evaluate the sub-challenges individually
        (N, k, _) = sub_challenges.shape
        evaluated_sub_challenges = super().ltf_eval(sub_challenges)

        # Duplicate the evaluation result for each vote and add individual noise
        # Note the votes are on the first axis
        evaled_inputs = broadcast_to(evaluated_sub_challenges, (self.vote_count, N, k))
        noise = self.random.normal(scale=self.sigma_noise, size=(self.vote_count, N, k))

        # Majority vote (i.e., sign(sum(·))) along the first axis
        return sign(np_sum(sign(evaled_inputs + noise), axis=0))


class XORArbiterPUF(NoisyLTFArray):
    """
    XOR Arbiter PUF. k Arbiter PUFs (so-called chains) are evaluated in parallel, the individual results are XORed and
    then returned.
    Devadas, S.: Physical unclonable functions (PUFS) and secure processors. In: Workshop on Cryptographic Hardware and
    Embedded Systems (2009)
    """

    @classmethod
    def transform_atf(cls, challenges: ndarray, k: int) -> ndarray:
        """
        Input transformation that simulates an Arbiter PUF
        :param challenges: array of shape(N,n)
                           Array of challenges which should be evaluated by the simulation.
        :param k: int
                  Number of LTFArray PUFs
        :return:  array of shape(N,k,n)
                  Array of transformed challenges, data type is same as input.
        """
        N, n = challenges.shape
        # TODO first copy, then ATT, then transform_id to enhance performance
        sub_challenges = cls.transform_id(challenges, 1).copy()
        cls.att(sub_challenges)
        return transpose(broadcast_to(transpose(sub_challenges, axes=(1, 0, 2)), (k, N, n)), axes=(1, 0, 2))

    def __init__(self, n: int, k: int, seed: int = None, transform: Union[Callable, str] = None,
                 noisiness: float = 0) -> None:
        if seed is None:
            seed = 'default'
        weight_seed = self.seed(f'xor arbiter puf {seed} weights')
        noise_seed = self.seed(f'xor arbiter puf {seed} noise')
        super().__init__(
            weight_array=self.normal_weights(n=n, k=k, seed=weight_seed),
            transform=transform or self.transform_atf,
            combiner=self.combiner_xor,
            sigma_noise=self.sigma_noise_from_random_weights(
                n=n,
                sigma_weight=1,
                noisiness=noisiness,
            ),
            seed=noise_seed,
        )

    def chain(self, idx: int) -> Simulation:
        r"""
        Returns a ``Simulation`` instance an arbiter chain of this XOR Arbiter PUF.
        :param idx: Index of the desired arbiter chain in :math:`\{0, ..., k\}`
        """
        if idx >= self.weight_array.shape[0]:
            raise IndexError
        return LTFArray(
            weight_array=self.weight_array[idx:idx + 1, :-1],
            transform=self.transform,
            combiner=self.combiner,
            bias=self.weight_array[idx:idx + 1, -1],
        )


class FeedForwardArbiterPUF(NoisyLTFArray):
    """
    Feed-Forward Arbiter PUF [GLCDD04]_. Simulation based on the additive delay model.
    """

    def __init__(self, n: int, ff: List[Tuple[int, int]], seed: int = None,
                 noisiness: float = 0) -> None:
        """
        Initialize a Feed-Forward Arbiter PUF Simulation.

        :param n: Number of challenge bits.

        :param ff: List of forward connections in the Feed-Forward Arbiter PUFs. Forward connections are defined by
            two-tuples :math:`(i,j)`, where :math:`i` defines the *arbiter position*, and :math:`j` defines the
            *feed position* of the feed forward loop. For each feed forward loop given, an arbiter element is simulated
            after the :math:`i`-th stage, with the result inserted as the challenge bit to the :math:`j`-th stage. Note
            that the Feed-Forward Arbiter PUF has `n + len(ff)` stages. It is required that :math:`i < j`, and no feed
            position may appear more than once per Arbiter PUF, however, it is allowed to use the arbiter position
            multiple times.

        :param seed: Seed for random weight generation.

        :param noisiness: Noise-level of the simulation.

        """
        if seed is None:
            seed = 'default'
        weight_seed = self.seed(f'FeedForwardArbiterPUF {seed} weights')
        noise_seed = self.seed(f'FeedForwardArbiterPUF {seed} noise')
        self.noise_prng = default_rng(noise_seed)
        self.ff = ff
        self.noisiness = noisiness

        super().__init__(
            weight_array=self.normal_weights(n=n + len(ff), k=1, seed=weight_seed),
            transform=XORArbiterPUF.transform_atf,
            combiner=self.combiner_xor,
            sigma_noise=self.sigma_noise_from_random_weights(
                n=n + len(ff),
                sigma_weight=1,
                noisiness=noisiness,
            ),
            seed=noise_seed,
        )

    @property
    def challenge_length(self) -> int:
        return self.weight_array.shape[1] - 1 - len(self.ff)

    n = challenge_length

    def val(self, challenges: ndarray) -> ndarray:
        (N, n) = challenges.shape
        ff = sorted(self.ff, key=lambda loop: loop[1])  # loops sorted by feed point
        feed_points = {feed_point for _, feed_point in ff}  # set of all feed points

        # prepare challenges with zeros for where the feed forward bits will go in
        ff_challenges = np.zeros(shape=(N, self.n + len(ff)))
        o = 0  # number of feed points before i-th stage
        for i in range(self.n + len(ff)):
            # set challenge bit but omit feed points (they will be filled below)
            if i in feed_points:
                o += 1
            else:
                ff_challenges[:, i] = challenges[:, i - o]

        # compute delay difference at each arbiter point and fill the sign into the corresponding feed point
        delay_difference = np.zeros(shape=(N,))
        delay_difference_pos = 0
        for arbiter_point, feed_point in ff + [(n + len(ff), None)]:

            # construct LTF array for the section from delay_difference_pos to the next arbiter point
            partial_puf = NoisyLTFArray(
                weight_array=self.weight_array[:, delay_difference_pos:arbiter_point],
                transform=self.transform,
                combiner=self.combiner,
                sigma_noise=self.sigma_noise_from_random_weights(
                    n=(arbiter_point or n) - delay_difference_pos,
                    sigma_weight=1,
                    noisiness=self.noisiness,
                ),
                seed=self.noise_prng.integers(2**32),
            )

            # select section of applied challenge
            ff_challenges_section = ff_challenges[:, delay_difference_pos:arbiter_point]
            assert 0 not in np.unique(ff_challenges_section)  # confirm all challenge bits are known

            # add delay difference of this section to grand total
            # The simulation is done using the **weights** of the Arbiter PUF, not the delays. This allows for
            # fast computation.
            delay_difference *= np.prod(ff_challenges_section, axis=1)
            delay_difference += partial_puf.ltf_eval(self.transform(ff_challenges_section, 1))[:, 0]
            delay_difference_pos = arbiter_point

            # except for the last iteration, set the resulting feed-forward challenge bit
            if feed_point:
                ff_challenges[:, feed_point] = sign(delay_difference)

        # note that we only take the sign in the eval method
        return delay_difference


class XORFeedForwardArbiterPUF(XORPUF):
    """
    XOR Feed-Forward Arbiter PUF [GLCDD04]_. Simulation based on the additive delay model.
    """

    def __init__(self, n: int, k: int, ff: Union[List[List[Tuple[int, int]]], List[Tuple[int, int]]], seed: int = None,
                 noisiness: float = 0) -> None:
        """
        Initialize an XOR Feed-Forward Arbiter PUF Simulation.

        :param n: Number of challenge bits.

        :param k: Number of Feed-Forward PUFs in this XOR Feed-Forward Arbiter PUF.

        :param ff: List of ``k`` lists of forward connections in the Feed-Forward Arbiter PUF. The :math:`l`-th Arbiter
            PUF in the XOR Feed-Forward Arbiter PUF will use the :math:`l`-th list of forward connections. If a list of
            tuples is given instead, it is used for all ``k`` Feed-Forward Arbiter PUFs. See
            :meth:`pypuf.simulation.FeedForwardArbiterPUF.__init__` for how to define feed-forward connections.

        :param seed: Seed for random weight generation.

        :param noisiness: Noise-level of the simulation.
        """
        if seed is None:
            seed = 'default'

        if not ff:
            ff = [[]]
        if ff[0] == [] or isinstance(ff[0], tuple):
            ff = [ff] * k

        ff_pufs = [
            FeedForwardArbiterPUF(
                n=n,
                ff=ff[l],
                seed=self.seed(f"XORFeedForwardArbiterPUF {seed} {l}"),
                noisiness=noisiness,
            )
            for l in range(k)
        ]

        super().__init__(ff_pufs)


class ArbiterPUF(XORArbiterPUF):

    def __init__(self, n: int, seed: int = None, transform: Union[Callable, str] = None, noisiness: float = 0) -> None:
        super().__init__(n, 1, seed, transform, noisiness)


class LightweightSecurePUF(XORArbiterPUF):
    """
    Lightweight Secure PUF. The challenge is transformed using a distinct pattern before the underlying XOR Arbiter PUF
    is evaluated on the transformed challenge. The result is returned.
    M. Majzoobi, F. Koushanfar, and M. Potkonjak, "Lightweight secure pufs", in IEEE/ACM International Conference on
    Computer-Aided Design (ICCAD 2008).
    """

    @classmethod
    def transform_shift(cls, challenges: ndarray, k: int) -> ndarray:
        """
        Input transformation that shifts the input bits for each of the k PUFs.
        :param challenges: array of shape(N,n)
                           Array of challenges which should be evaluated by the simulation.
        :param k: int
                  Number of LTFArray PUFs
        :return:  array of shape(N,k,n)
                  Array of transformed challenges.
        """
        N = len(challenges)
        n = len(challenges[0])

        result = swapaxes(array([
            concatenate((challenges[:, shift_by:], challenges[:, :shift_by]), axis=1)
            for shift_by in range(k)
        ]), 0, 1)

        assert result.shape == (N, k, n)
        return result

    @classmethod
    def transform_lightweight_secure(cls, challenges: ndarray, k: int) -> ndarray:
        """
        Input transform as defined by Majzoobi et al. 2008.
        :param challenges: array of shape(N,n)
                           Array of challenges which should be evaluated by the simulation.
        :param k: int
                  Number of LTFArray PUFs
        :return:  array of shape(N,k,n)
                  Array of transformed challenges.
        """
        (N, n) = challenges.shape
        assert n % 2 == 0, 'Secure Lightweight Input Transformation only defined for even n.'

        sub_challenges = cls.transform_shift(challenges, k)

        sub_challenges = transpose(
            concatenate(
                (
                    [sub_challenges[:, :, i] * sub_challenges[:, :, i + 1] for i in range(0, n, 2)],
                    [sub_challenges[:, :, 0]],
                    [sub_challenges[:, :, i] * sub_challenges[:, :, i + 1] for i in range(1, n - 2, 2)],
                )
            ),
            (1, 2, 0)
        )

        cls.att(sub_challenges)

        assert sub_challenges.shape == (N, k, n), 'The resulting challenges do not have the desired shape.'
        return sub_challenges

    def __init__(self, n: int, k: int, seed: int = None, noisiness: float = 0) -> None:
        super().__init__(n, k, seed, self.transform_lightweight_secure, noisiness)


class RandomTransformationPUF(XORArbiterPUF):

    @classmethod
    def transform_random(cls, challenges: ndarray, k: int) -> ndarray:
        """
        This input transformation chooses for each Arbiter Chain an random challenge based on the initial challenge.
        :param challenges: array of shape(N,n)
                           Array of challenges which should be evaluated by the simulation.
        :param k: int
                  Number of LTFArray PUFs
        :return:  array of shape(N,k,n)
                  Array of transformed challenges.
        """
        (N, n) = challenges.shape

        cs_01 = copy(challenges)
        cs_01[cs_01 == 1] = 0
        cs_01[cs_01 == -1] = 1

        result = array([RandomState(c).choice((-1, 1), (k, n)) for c in cs_01], dtype=challenges.dtype)

        assert result.shape == (N, k, n), 'The resulting challenges have not the desired shape.'
        return result

    def __init__(self, n: int, k: int, seed: int = None, noisiness: float = 0) -> None:
        super().__init__(n, k, seed, self.transform_random, noisiness)


class PermutationPUF(XORArbiterPUF):

    FIXED_PERMUTATION_SEEDS = {
        16: [2989, 2992, 3063, 3068, 3842, 4128, 5490, 24219, 144312, 469287],
        18: [2995, 3013, 3035, 3152, 3232, 4722, 6679, 15571, 523353, 2397496],
        20: [2995, 3028, 3033, 3066, 3075, 3689, 15194, 54679, 81174, 395475],
        22: [2989, 3013, 3026, 3273, 3382, 4551, 5381, 14403, 81466, 179641],
        24: [2994, 2996, 3028, 3030, 3160, 4884, 6245, 6474, 87312, 194512],
        26: [2989, 3013, 3015, 3051, 3264, 3302, 3429, 18289, 127895, 279478],
        28: [2991, 2994, 2996, 3037, 3139, 3605, 3861, 15627, 62080, 143257],
        30: [2991, 2994, 3015, 3035, 3283, 6952, 8703, 20957, 59710, 65892],
        32: [2991, 2994, 3063, 3115, 3316, 4131, 10269, 11689, 46468, 165660],
        34: [2993, 2997, 3004, 3120, 3471, 4064, 8774, 20811, 61247, 93418],
        36: [2991, 2995, 3009, 3033, 3749, 4266, 9237, 10500, 18740, 26595],
        38: [2989, 2995, 2996, 3079, 3173, 3724, 6873, 12350, 15520, 205407],
        40: [2990, 2997, 3014, 3019, 3183, 3476, 11754, 15055, 25355, 130679],
        42: [2989, 3009, 3024, 3062, 3079, 3630, 3716, 7020, 42035, 350609],
        44: [2990, 2995, 2999, 3029, 3458, 4807, 7088, 8618, 83012, 196591],
        46: [2989, 3013, 3033, 3064, 3310, 3523, 4672, 7600, 16544, 43549],
        48: [2989, 3005, 3019, 3038, 3395, 3881, 4232, 4477, 8335, 77849],
        50: [2989, 2996, 3005, 3097, 3304, 4456, 7273, 15858, 31149, 104186],
        52: [2990, 3015, 3057, 3068, 3301, 4099, 5633, 9482, 20989, 310225],
        54: [2990, 2993, 3002, 3011, 3024, 3972, 4029, 4499, 17427, 47736],
        56: [2989, 3002, 3008, 3026, 3044, 3130, 4032, 5897, 14225, 65043],
        58: [2989, 2992, 3008, 3210, 3283, 4146, 4484, 5782, 64291, 70967],
        60: [2991, 3006, 3028, 3034, 3270, 3631, 6567, 8517, 11912, 30874],
        62: [2990, 3002, 3006, 3230, 3234, 3542, 6541, 11357, 13089, 26552],
        64: [2989, 2992, 3038, 3084, 3457, 6200, 7089, 18369, 21540, 44106],
        66: [2990, 2995, 3006, 3067, 3093, 3370, 4794, 6734, 25049, 48708],
        68: [2989, 2990, 3024, 3038, 4245, 4363, 4732, 16659, 18994, 66883],
        70: [2995, 3003, 3009, 3026, 3146, 4787, 4818, 9204, 13807, 94710],
        72: [2989, 2998, 3002, 3045, 3627, 4051, 6324, 33183, 37736, 60692],
        74: [2989, 2993, 3004, 3125, 3175, 4275, 4536, 9275, 11050, 40355],
        76: [2995, 3006, 3017, 3028, 3042, 3272, 3537, 8448, 12202, 13195],
        78: [2989, 2990, 3019, 3079, 3103, 3351, 3398, 11866, 13086, 32017],
        80: [2989, 3011, 3019, 3078, 3201, 3418, 3724, 9987, 11980, 27793],
        82: [2989, 2992, 3020, 3107, 3401, 3717, 8509, 9415, 12150, 25195],
        84: [2989, 3003, 3036, 3056, 3167, 3472, 6306, 10187, 21982, 30975],
        86: [2989, 2991, 3049, 3077, 3145, 5066, 8076, 16926, 21304, 39299],
        88: [2989, 2996, 3028, 3122, 3123, 3339, 6654, 11236, 17009, 22022],
        90: [2993, 3009, 3012, 3035, 3060, 3358, 3888, 4505, 6559, 30776],
        92: [2995, 2997, 3018, 3150, 3239, 3441, 3522, 6798, 13129, 123838],
        94: [2989, 2997, 3000, 3078, 3320, 3486, 4200, 9943, 12646, 36760],
        96: [2990, 2994, 2996, 3110, 3388, 3877, 4405, 8225, 9588, 39426],
        98: [2993, 3000, 3021, 3108, 3151, 4409, 5928, 10655, 16362, 41790],
        100: [2991, 2995, 3039, 3044, 3093, 3507, 4209, 12733, 25474, 87190],
        102: [2990, 3005, 3031, 3102, 3129, 3178, 5126, 11292, 45648, 49704],
        104: [2991, 2999, 3019, 3069, 3391, 4022, 4786, 5277, 48976, 54710],
        106: [2990, 2992, 2999, 3111, 3223, 3409, 4137, 7802, 17399, 33181],
        108: [2992, 2994, 2995, 3006, 3025, 3795, 3797, 8635, 10014, 14284],
        110: [2991, 3017, 3025, 3075, 3289, 3496, 4500, 5454, 16451, 32577],
        112: [2992, 3003, 3006, 3021, 3509, 3673, 4005, 5394, 17151, 26814],
        114: [2989, 3025, 3055, 3097, 3185, 3558, 3614, 5636, 9899, 20402],
        116: [2989, 2995, 3024, 3043, 3065, 3087, 4382, 5041, 31225, 51708],
        118: [2990, 2992, 3000, 3012, 3434, 3560, 4907, 6905, 11512, 34843],
        120: [2989, 2992, 3006, 3017, 3167, 3954, 4666, 7778, 19259, 63831],
        122: [2992, 3003, 3005, 3036, 3126, 3204, 4767, 4968, 13048, 19272],
        124: [2989, 2991, 3023, 3052, 3238, 4002, 5143, 8080, 9957, 20451],
        126: [2989, 2991, 3015, 3113, 3125, 3344, 3409, 11948, 14555, 31180],
        128: [2989, 3006, 3009, 3031, 3437, 4174, 6045, 7906, 11554, 29544],
        130: [2991, 3012, 3057, 3167, 3265, 3293, 3586, 4726, 11488, 17077],
        132: [2992, 3005, 3007, 3022, 3159, 5420, 5711, 12826, 37745, 46518],
        134: [2995, 3021, 3041, 3104, 3367, 3467, 4041, 8165, 9130, 37357],
        136: [2990, 3008, 3079, 3108, 3612, 3846, 5110, 8754, 9618, 19755],
        138: [2989, 3005, 3034, 3099, 3317, 3732, 4603, 8624, 11293, 50793],
        140: [2990, 2994, 3018, 3026, 3362, 3786, 5453, 10367, 47083, 56850],
        142: [2992, 2993, 3019, 3045, 3068, 4530, 6105, 9011, 23005, 69150],
        144: [2992, 2998, 3028, 3035, 3145, 3633, 5509, 7077, 11903, 45726],
        146: [2989, 2996, 2997, 3044, 3092, 3503, 5325, 7504, 11578, 65981],
        148: [2989, 2992, 3029, 3044, 3068, 3167, 7129, 8330, 9828, 15322],
        150: [2989, 3012, 3068, 3174, 3404, 3420, 3579, 11735, 12326, 20612],
        152: [2994, 3025, 3037, 3051, 3307, 3686, 4584, 7950, 14806, 69770],
        154: [2989, 3006, 3017, 3126, 3349, 4360, 7409, 12323, 34353, 52056],
        156: [2989, 2990, 3022, 3035, 3189, 3309, 4969, 8641, 12876, 77672],
        158: [2990, 2991, 3006, 3040, 3068, 3154, 4052, 4252, 9121, 53327],
        160: [2991, 2992, 3007, 3070, 3077, 3352, 3414, 10070, 20332, 36424],
        162: [2991, 2994, 3015, 3203, 3236, 3433, 3586, 3873, 4784, 50400],
        164: [2990, 3011, 3041, 3061, 3167, 3898, 7766, 10819, 24894, 36605],
        166: [2990, 2996, 3047, 3161, 3175, 3342, 4140, 9229, 11541, 98858],
        168: [2990, 2995, 2997, 3105, 3221, 3322, 5948, 6340, 16584, 20705],
        170: [2989, 3002, 3008, 3019, 3138, 4020, 5437, 6867, 16346, 71410],
        172: [2989, 3003, 3010, 3034, 3084, 3170, 3530, 8196, 60195, 104186],
        174: [2989, 2995, 2996, 3007, 3024, 3106, 3751, 4447, 4837, 47197],
        176: [2991, 2993, 3004, 3108, 3539, 3778, 5034, 5323, 11872, 82653],
        178: [2989, 2990, 3044, 3075, 3098, 3173, 11281, 14674, 17103, 60624],
        180: [2989, 2995, 3007, 3013, 3029, 4109, 6162, 7353, 10539, 44903],
        182: [2991, 3011, 3035, 3039, 3150, 3574, 4658, 5161, 12661, 50586],
        184: [2989, 2995, 3005, 3015, 3096, 3244, 4601, 5185, 18873, 34847],
        186: [2995, 3002, 3032, 3127, 3229, 3638, 4014, 6884, 27991, 92847],
        188: [2992, 3005, 3011, 3101, 3120, 3152, 3224, 7025, 26182, 35634],
        190: [2992, 2997, 3004, 3067, 3313, 3481, 4345, 7488, 17196, 34975],
        192: [2996, 3006, 3037, 3139, 3380, 3895, 5390, 5460, 20473, 81144],
        194: [2990, 2998, 3049, 3104, 3141, 5570, 7041, 7973, 16742, 58891],
        196: [2991, 3000, 3014, 3041, 3054, 4667, 5857, 9591, 9991, 57900],
        198: [2989, 2995, 3001, 3030, 3159, 3361, 3717, 5744, 18435, 35850],
        200: [2989, 2990, 3009, 3090, 3192, 3395, 5175, 12545, 26908, 80719],
        202: [2991, 3004, 3047, 3110, 3115, 3199, 4379, 5564, 14899, 20695],
        204: [2989, 2992, 2997, 3043, 3704, 3988, 4261, 9209, 22444, 37516],
        206: [2992, 3013, 3019, 3120, 3275, 4259, 5381, 9168, 17342, 19600],
        208: [2993, 2995, 3000, 3072, 3164, 4300, 5019, 10519, 11634, 18463],
        210: [2996, 3002, 3084, 3114, 3169, 3319, 4021, 9534, 9572, 13347],
        212: [2989, 3003, 3048, 3102, 3348, 3703, 5611, 8864, 40666, 40954],
        214: [2991, 3000, 3011, 3082, 3200, 3366, 4677, 6348, 30148, 66127],
        216: [2989, 2993, 3018, 3086, 3088, 3358, 3841, 4289, 14258, 36230],
        218: [2992, 3000, 3017, 3064, 3113, 3133, 6216, 8532, 31936, 39415],
        220: [2989, 2995, 3003, 3039, 3535, 3745, 4926, 5587, 11947, 26910],
        222: [2989, 2995, 3021, 3121, 3427, 3571, 5404, 6172, 27491, 31231],
        224: [2989, 2992, 2994, 3109, 3183, 3561, 5388, 5656, 10226, 24444],
        226: [2990, 3014, 3038, 3049, 3956, 4268, 4818, 4946, 9262, 75477],
        228: [2993, 3000, 3012, 3046, 3067, 3211, 3737, 4212, 12529, 37219],
        230: [2997, 3008, 3016, 3250, 3329, 3662, 3905, 4826, 34195, 45328],
        232: [2991, 2997, 2999, 3022, 3130, 3211, 3713, 4570, 6566, 11935],
        234: [2994, 2999, 3006, 3061, 3175, 3215, 3315, 3642, 14245, 34051],
        236: [2990, 3006, 3022, 3201, 3282, 3587, 3602, 5818, 17394, 29636],
        238: [2990, 2997, 3008, 3040, 3335, 3777, 4014, 12314, 51708, 110648],
        240: [2990, 2999, 3042, 3066, 3550, 3820, 4196, 4333, 23161, 34698],
        242: [2994, 3001, 3004, 3184, 3324, 3432, 3739, 9498, 9561, 13463],
        244: [2989, 2995, 3002, 3004, 3122, 3247, 4197, 7187, 17291, 24364],
        246: [2995, 3008, 3059, 3115, 3354, 3545, 3977, 17567, 31344, 102089],
        248: [2989, 2991, 3002, 3014, 3062, 3676, 3845, 12201, 13862, 15246],
        250: [2989, 3016, 3073, 3074, 3524, 3915, 4614, 10463, 12071, 13276],
        252: [2989, 2997, 3017, 3095, 3202, 3420, 8478, 23842, 24246, 47867],
        254: [2990, 3008, 3032, 3209, 3722, 4619, 5307, 8650, 15598, 16398],
        256: [2991, 3009, 3016, 3091, 3243, 4669, 5259, 5691, 8253, 22722],
    }

    @classmethod
    def transform_fixed_permutation(cls, challenges: ndarray, k: int) -> ndarray:
        """
        Permutes the challenge bits using hardcoded, fix point free permutations designed such that no
        sub-challenge bit gets permuted equally for all other generated sub-challenges. Such permutations
        are not easy to find, hence this function only supports a limited number of n and k.
        After permutation, we apply the ATF transform to the sub-challenges (hence generating what in the
        linear Arbiter PUF model is called feature vectors).
        :param challenges: array of shape(N,n)
                           Array of challenges which should be evaluated by the simulation.
        :param k: int
                  Number of LTFArray PUFs
        :return:  array of shape(N,k,n)
                  Array of transformed challenges.
        """

        # check parameter n
        # For performance reasons, we do not call self._find_fixed_permutations here.
        n = len(challenges[0])
        assert n in cls.FIXED_PERMUTATION_SEEDS.keys(), \
            'Fixed permutation currently not supported for n=%i, but only for n in %s. ' \
            'To add support, please use LTFArray._find_fixed_permutations(n, k).' % \
            (n, cls.FIXED_PERMUTATION_SEEDS.keys())

        # check parameter k
        seeds = cls.FIXED_PERMUTATION_SEEDS[n]
        assert k <= len(seeds), 'Fixed permutation for n=%i currently only supports k<=%i.' % (n, len(seeds))

        # generate permutations (use legacy random generator for consistency)
        permutations = [RandomState(seed).permutation(n) for seed in seeds]

        # perform permutations
        result = swapaxes(
            array([
                challenges[:, permutations[i]]
                for i in range(k)
            ], dtype=challenges.dtype),
            0,
            1
        )

        cls.att(result)
        return result

    @classmethod
    def _find_fixed_permutations(cls, n: int, k: int) -> List[int]:
        """
        Finds permutations suitable to use in the Permutation PUF

        Permutations are chosen such that no permutation has a fix point and no
        two permutations share at least one point. (See `permutation_okay` below.)

        Note that the run time of this method increases drastically with k. On an
        Intel i7, n=64, k=10 takes a couple of seconds.

        :return: list of seeds for `RandomState`. Obtain the permutation with
          `RandomState(seed).permutation(n)`.
        """
        def permutation_okay(new_p: ndarray, ps: List[ndarray]) -> bool:
            """ returns True if the permutation qualifies """

            # 1. check that p has no fix point
            if any([i == new_p[i] for i in range(len(new_p))]):
                return False

            # 2. check that it does not share a point if any old_p in ps:
            if any([
                    any([old_p[i] == new_p[i] for i in range(len(new_p))])
                    for old_p in ps
            ]):
                return False

            return True

        seed = 0xbad
        permutation_seeds = []
        permutations = []

        while len(permutations) < k:
            prng = RandomState(seed)
            p = prng.permutation(n)
            if permutation_okay(p, permutations):
                permutation_seeds.append(seed)
                permutations.append(p)
            seed += 1

        return permutation_seeds

    def __init__(self, n: int, k: int, seed: int = None, noisiness: float = 0) -> None:
        super().__init__(n, k, seed, self.transform_fixed_permutation, noisiness)


class InterposePUF(Simulation):
    """
    Interpose PUF. Essentially consisting of two XOR Arbiter PUFs, where the second XOR Arbiter PUF has challenge length
    n+1. The value of the middle challenge bit is the result bit of the first XOR Arbiter PUF.
    Phuong Ha Nguyen, Durga Prasad Sahoo, Chenglu Jin, Kaleel Mahmood, Ulrich Rührmair and Marten van Dijk,
    "The Interpose PUF: Secure PUF Design against State-of-the-art Machine Learning Attacks", CHES 2019.
    """

    def __init__(self, n: int, k_down: int, k_up: int = 1, interpose_pos: int = None, seed: int = None,
                 noisiness: float = 0) -> None:
        super().__init__()
        if seed is None:
            seed = 'default'
        seed_up = self.seed(f'interpose puf {seed} up')
        seed_down = self.seed(f'interpose puf {seed} down')
        self.up = XORArbiterPUF(n, k_up, seed_up, XORArbiterPUF.transform_atf, noisiness)
        self.down = XORArbiterPUF(n + 1, k_down, seed_down, XORArbiterPUF.transform_atf, noisiness)
        self.interpose_pos = interpose_pos or n // 2

    @property
    def challenge_length(self) -> int:
        return self.up.challenge_length

    @property
    def response_length(self) -> int:
        return self.down.response_length

    def _interpose_bits(self, challenges: ndarray) -> ndarray:
        (N, _) = challenges.shape
        return self.up.eval(challenges).reshape(N, 1)

    def eval(self, challenges: ndarray) -> ndarray:
        (N, n) = challenges.shape
        interpose_bits = self._interpose_bits(challenges)
        down_challenges = concatenate(
            (challenges[:, :self.interpose_pos], interpose_bits, challenges[:, self.interpose_pos:]),
            axis=1
        )
        assert down_challenges.shape == (N, n + 1)
        return self.down.eval(down_challenges)
