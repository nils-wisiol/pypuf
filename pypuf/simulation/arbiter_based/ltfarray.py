"""
This module provides several different implementations of arbiter PUF simulations. The linear threshold function array
model is the core of each simulation class.
"""
from numpy import prod, shape, sign, array, transpose, concatenate, swapaxes, sqrt, amax, append
from numpy import sum as np_sum, ones, ndarray, zeros, reshape, broadcast_to, einsum
from numpy.random import RandomState

from pypuf import tools
from pypuf.simulation.base import Simulation


class CompoundTransformation:
    """
    Defines an input transformation that is build from a generator function.
    The purpose of this class is mainly to define generated input transformations in a way they can be pickled.
    """

    def __init__(self, generator, args):
        """
        Defines a compound input transformation
        :param generator: Generator function to be used.
        :param args: Arguments for the generator function.
        """
        self.generator = generator
        self.args = args
        self.__name__ = generator(*args).__name__
        self._transform = None

    def build(self):
        """
        Calls the generator function, supplies the defined arguments and returns the result
        :return: The compound input transformation as defined for this instance.
        """
        return self.generator(*self.args)

    def __call__(self, *args, **kwargs):
        """
        If called directly, we build and cache the compound transformation. Hence, this class
        behaves transparent if treated directly as input transformation with few overhead.
        """
        if not self._transform:
            self._transform = self.build()
        return self._transform(*args, **kwargs)

    def __repr__(self):
        return self.build().__name__


class LTFArray(Simulation):
    """
    Class that simulates k LTFs with n bits and a constant term each
    and constant bias added.
    """

    @classmethod
    def combiner_xor(cls, responses):
        """
        combines output responses with the XOR operation
        :param responses: Array of int of float with shape(N,k,n)
                          An Array with a number of vectors of single LTF results
        :return: array of float or int shape(N)
                 Array of responses for the N different challenges.
        """
        return prod(responses, axis=1)

    @classmethod
    def combiner_ip_mod2(cls, responses):
        """
        combines output responses with the inner product mod 2 operation
        :param responses: a array with a number of vectors of single LTF results
        :return: array of float or int shape(N)
                 Array of responses for the N different challenges.
        """
        n = len(responses[0])
        assert n % 2 == 0, 'IP mod 2 is only defined for even n.'
        return prod(
            transpose(
                [
                    amax((responses[:, i], responses[:, i + 1]), 0)
                    for i in range(0, n, 2)
                ]),
            1
        )

    @classmethod
    def transform_id(cls, challenges, k):
        """
        Input transformation that does nothing.
        :param challenges: array of shape(N,n)
                           Array of challenges which should be evaluated by the simulation.
        :param k: int
                  Number of LTFArray PUFs
        :return:  array of shape(N,k,n)
                  Array of transformed challenges.
        """
        (N, n) = challenges.shape
        return transpose(broadcast_to(challenges, (k, N, n)), axes=(1, 0, 2))

    @classmethod
    def transform_atf(cls, challenges, k):
        """
        Input transformation that simulates an Arbiter PUF
        :param challenges: array of shape(N,n)
                           Array of challenges which should be evaluated by the simulation.
        :param k: int
                  Number of LTFArray PUFs
        :return:  array of shape(N,k,n)
                  Array of transformed challenges, data type is same as input.
        """
        # Transform with ATF monomials
        dtype = challenges.dtype
        challenges = transpose(
            array([
                prod(challenges[:, i:], 1, dtype=dtype)
                for i in range(len(challenges[0]))
            ], dtype=dtype)
        )

        # Same challenge for all k Arbiters
        res = cls.transform_id(challenges, k)
        return res

    @classmethod
    def transform_lightweight_secure(cls, challenges, k):
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

        result = cls.att(sub_challenges)

        assert result.shape == (N, k, n), 'The resulting challenges do not have the desired shape.'
        return result

    @classmethod
    def transform_soelter_lightweight_secure(cls, challenges, k):
        """
        Input transformation like defined by Majzoobi et al. (cf. transform_lightweight_secure),
        but differs in one bit. Introduced by Sölter.
        :param challenges: array of shape(N,n)
                           Array of challenges which should be evaluated by the simulation.
        :param k: int
                  Number of LTFArray PUFs
        :return:  array of shape(N,k,n)
                  Array of transformed challenges.
        """
        (N, n) = challenges.shape
        assert n % 2 == 0, 'Sölter\'s Secure Lightweight Input Transformation only defined for even n.'
        n_half = int(n / 2)

        challenges = transpose(
            concatenate(
                (
                    [challenges[:, i] * challenges[:, i + 1] for i in range(0, n, 2)],  # (x1x2, x3x4, ... xn-1xn)
                    [challenges[:, n_half]],  # (x_(n/2+1))
                    [challenges[:, i] * challenges[:, i + 1] for i in range(1, n - 2, 2)],  # (x2x3, x4x5, ... xn-2xn-1)
                )
            )
        )

        assert challenges.shape == (N, n)
        res = cls.transform_shift(challenges, k)
        return res

    @classmethod
    def transform_shift(cls, challenges, k):
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
            concatenate((challenges[:, l:], challenges[:, :l]), axis=1)
            for l in range(k)
        ]), 0, 1)

        assert result.shape == (N, k, n)
        return result

    @classmethod
    def transform_polynomial(cls, challenges, k):
        """
        This input transformation interprets a challenge c as a
        polynomial over the finite field GF(2^n)=F2/f*F2, where f is a
        irreducible polynomial of degree n.
        The irreducible polynomial f is hard coded and
        of degree 8, 16, 24, 32, 48, or 64.
        Each Arbiter Chain i receives as input the polynomial c^i
        as element of GF(2^n).
        :param challenges: array of shape(N,n)
                           Array of challenges which should be evaluated by the simulation.
        :param k: int
                  Number of LTFArray PUFs
        :return:  array of shape(N,k,n)
                  Array of transformed challenges.
        """
        dtype = challenges.dtype
        N = len(challenges)
        n = len(challenges[0])

        if n == 64:
            irreducible_polynomial = array(
                [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                dtype=dtype
            )
        elif n == 48:
            irreducible_polynomial = array(
                [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1], dtype=dtype
            )
        elif n == 32:
            irreducible_polynomial = array(
                [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                dtype=dtype
            )
        elif n == 24:
            irreducible_polynomial = array([1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                           dtype=dtype)
        elif n == 16:
            irreducible_polynomial = array([1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1], dtype=dtype)
        elif n == 8:
            irreducible_polynomial = array([1, 0, 1, 0, 0, 1, 1, 0, 1], dtype=dtype)
        else:
            raise AssertionError('Polynomial transformation is only implemented for challenges with n in '
                                 '{8, 16, 24, 32, 48, 64}.')

        # Transform challenge to 0,1 array to compute transformation with numpy.
        cs_01 = array([tools.transform_challenge_11_to_01(c) for c in challenges])

        # Compute c^i for each challenge for i from 1 to k.
        challenges = concatenate([
            [tools.poly_mult_div(c, irreducible_polynomial, k) for c in cs_01]
        ])

        # Transform challenges back to -1,1 notation.
        result = array([tools.transform_challenge_01_to_11(c) for c in challenges], dtype=dtype)

        assert result.shape == (N, k, n), 'The resulting challenges have not the desired shape.'
        return result

    @classmethod
    def transform_permutation_atf(cls, challenges, k):
        """
        This transformation performs first a pseudorandom permutation of the challenge k times before applying the
        ATF transformation to each challenge.
        :param challenges: array of shape(N,n)
                           Array of challenges which should be evaluated by the simulation.
        :param k: int
                  Number of LTFArray PUFs
        :return:  array of shape(N,k,n)
                  Array of transformed challenges.
        """
        N = len(challenges)
        n = len(challenges[0])
        seed = 0x1234

        # Perform random permutations
        sub_challenges = array(
            [
                [RandomState(seed + i).permutation(c)
                 for i in range(k)]
                for c in challenges
            ]
        )
        result = cls.att(sub_challenges)

        assert result.shape == (N, k, n), 'The resulting challenges have not the desired shape.'
        return result

    @classmethod
    def transform_random(cls, challenges, k):
        """
        This input transformation chooses for each Arbiter Chain an random challenge based on the initial challenge.
        :param challenges: array of shape(N,n)
                           Array of challenges which should be evaluated by the simulation.
        :param k: int
                  Number of LTFArray PUFs
        :return:  array of shape(N,k,n)
                  Array of transformed challenges.
        """
        N = len(challenges)
        n = len(challenges[0])

        cs_01 = array([tools.transform_challenge_11_to_01(c) for c in challenges], dtype=challenges.dtype)

        result = array([RandomState(c).choice((-1, 1), (k, n)) for c in cs_01], dtype=challenges.dtype)

        assert result.shape == (N, k, n), 'The resulting challenges have not the desired shape.'
        return result

    FIXED_PERMUTATION_SEEDS = {
        8: [2990, 3002, 3017, 3044, 3236, 9300],
        10: [2989, 3001, 3015, 3032, 3091, 12880],
        12: [2989, 2990, 2995, 3030, 3586, 4234, 12838, 184995],
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
    def transform_fixed_permutation(cls, challenges, k):
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

        # generate permutations
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

        result = cls.att(result)

        return result

    @classmethod
    def _find_fixed_permutations(cls, n, k):
        """
        Finds permutations suitable to use in LTFArray.transform_fixed_permutation.

        Permutations are chosen such that no permutation has a fix point and no
        two permutations share at least one point. (See `permutation_okay` below.)

        Note that the run time of this method increases drastically with k. On an
        Intel i7, n=64, k=10 takes a couple of seconds.

        :return: list of seeds for `RandomState`. Obtain the permutation with
          `RandomState(seed).permutation(n)`.
        """
        def permutation_okay(new_p, ps):
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

    @classmethod
    def generate_stacked_transform(cls, transform_1, puf_count, transform_2):
        """
        Returns an input transformation that will transform the first puf_count challenges using transform_1,
        the remaining k - puf_count challenges using transform_2.
        :param transform_1: A function: array of int with shape(N,n), int number of PUFs k -> shape(N,k,n)
                            The function transforms input challenges in order to increase resistance against attacks.
        :param puf_count: int
                          Number of LTFs that receive challenges transformed with transform_1.
        :param transform_2: A function: array of int with shape(N,n), int number of PUFs k -> shape(N,k,n)
                            The function transforms input challenges in order to increase resistance against attacks.
        :return: A function: array of int with shape(N,n), int number of PUFs k -> shape(N,k,n)
                 A function that can perform the desired transformation.
        """

        def transform(challenges, k):
            """
           Method as described in generate_concatenated_transform doc string.
           :param challenges: array of shape(N,n)
                              Array of challenges which should be evaluated by the simulation.
           :param k: int
                     Number of LTFArray PUFs
           :return: A function: array of int with shape(N,n), int number of PUFs k -> shape(N,k,n)
                    A function that can perform the desired transformation.
           """
            (N, n) = challenges.shape
            transformed_1 = transform_1(challenges, puf_count)
            transformed_2 = transform_2(challenges, k - puf_count)
            assert transformed_1.shape == (N, puf_count, n)
            assert transformed_2.shape == (N, k - puf_count, n)
            return concatenate(
                (
                    transformed_1,
                    transformed_2,
                ),
                axis=1
            )

        transform.__name__ = 'transform_stack_%s_nn%i_%s' % \
                             (
                                 transform_1.__name__.replace('transform_', ''),
                                 puf_count,
                                 transform_2.__name__.replace('transform_', '')
                             )

        return transform

    @classmethod
    def generate_random_permutation_transform(cls, seed, nn, kk, atf=False):
        """
        Returns an input transformation that uses k pseudorandomly generated permutations
        :param seed: int
                     Seed for the pseudorandom generation
        :param nn: int Challenge length (must equal LTFArray.n)
        :param kk: int Number of permutations to be used (must equal LTFArray.k)
        :param atf: boolean
                    Perform ATF transform after permuting
        :return:  A function: array of int with shape(N,n), int number of PUFs k -> shape(N,k,n)
                  A function that can perform the desired transformation.
        """
        prng = RandomState(seed)
        permutations = [prng.permutation(nn) for _ in range(kk)]

        def transform(challenges, k):
            """
            Method as described in generate_concatenated_transform doc string.
            :param challenges: array of shape(N,n)
                               Array of challenges which should be evaluated by the simulation.
            :param k: int
                     Number of LTFArray PUFs
            :return: A function: array of int with shape(N,n), int number of PUFs k -> shape(N,k,n)
                     A function that can perform the desired transformation.
            """
            (_, n) = challenges.shape
            assert k == kk and n == nn, \
                'Permutations Input Transform cannot be used for LTFArrays with size other than defined'

            sub_challenges = swapaxes(
                array([
                    challenges[:, permutations[i]]
                    for i in range(kk)
                ]),
                0,
                1
            )

            if atf:
                # Perform atf transform
                sub_challenges = cls.att(sub_challenges)

            return sub_challenges

        transform.__name__ = 'transform_permutations' + ('_plus_atf_' if atf else '') + '_%x' % seed
        return transform

    @classmethod
    def generate_concatenated_transform(cls, transform_1, bit_number_transform_1, transform_2):
        """
        Returns an input transformation that will transform the first bit_number_transform_1 bit of each challenge using
        transform_1, the remaining bits using transform_2.
        :param transform_1: A function: array of int with shape(N,n), int number of PUFs k -> shape(N,k,n)
                            The function transforms input challenges in order to increase resistance against attacks.
        :param bit_number_transform_1: int
                                       Number of challenge bits which are to be transformed with transform_1
        :param transform_2: A function: array of int with shape(N,n), int number of PUFs k -> shape(N,k,n)
                            The function transforms input challenges in order to increase resistance against attacks.
        :return A function: array of int with shape(N,n), int number of PUFs k -> shape(N,k,n)
                A function that can perform the desired transformation.
        """

        def transform(challenges, k):
            """
            Method as described in generate_concatenated_transform doc string.
            :param challenges: array of shape(N,n)
                               Array of challenges which should be evaluated by the simulation.
            :param k: int
                      Number of LTFArray PUFs
            :return: A function: array of int with shape(N,n), int number of PUFs k -> shape(N,k,n)
                     A function that can perform the desired transformation.
            """
            (N, n) = challenges.shape
            challenges1 = challenges[:, :bit_number_transform_1]
            challenges2 = challenges[:, bit_number_transform_1:]
            transformed_1 = transform_1(challenges1, k)
            transformed_2 = transform_2(challenges2, k)
            assert transformed_1.shape == (N, k, bit_number_transform_1)
            assert transformed_2.shape == (N, k, n - bit_number_transform_1)
            return concatenate(
                (
                    transformed_1,
                    transformed_2
                ),
                axis=2
            )

        transform.__name__ = 'transform_concat_%s_nn%i_%s' % \
                             (
                                 transform_1.__name__.replace('transform_', ''),
                                 bit_number_transform_1,
                                 transform_2.__name__.replace('transform_', '')
                             )

        return transform

    @classmethod
    def att(cls, sub_challenges):
        """
        Performs the "Arbiter Threshold Transform" (ATT) on an array of sub-challenges.
        ATT is defined to modify any given sub-challenge c as follows:
        Let c be a vector of n bits, then the i-th output bit of ATT(c) equals
        prod_(j=i)^n c_j, i.e. the i-th output bit is the product of the i-th input bit
        and all following input bits.

        This method performs ATT in situ, i.e. without (much) additional memory.
        The input array will be overwritten.
        :param sub_challenges: array of shape (N, k, n), where N is the total number of
        sub-challenge tuples, k is the number of sub-challenges per master-challenge, and
        n is the number of bits per sub-challenge.
        :return: transformed array of sub-challenges, shape (N, k, n)
        """
        (_, _, n) = sub_challenges.shape
        for i in range(n - 2, -1, -1):
            sub_challenges[:, :, i] *= sub_challenges[:, :, i + 1]
        return sub_challenges

    @classmethod
    def att_inverse(cls, sub_challenges):
        """
        Performs the inverse "Arbiter Threshold Transform" (ATT) on an array of sub-challenges.
        The inverse ATT is defined to modify any given sub-challenge x as follows:
        Let x be a vector of n bits, then the i-th output bit of ATT_inverse(x) equals
        x_i / x_(i+1), where x_(n+1) is treated as 1. I.e. the i-th output bit is the division
        of the i-th input bit and the following input bit.

        This method performs ATT in situ, i.e. without (much) additional memory.
        The input array will be overwritten.

        This method is defined for input bits in {-1,1} only, using other bits has undefined
        behavior.

        :param sub_challenges: array of shape (N, k, n), where N is the total number of
        sub-challenge tuples, k is the number of sub-challenges per master-challenge, and
        n is the number of bits per sub-challenge.
        :return: transformed array of sub-challenges, shape (N, k, n)
        """
        (_, _, n) = sub_challenges.shape
        for i in range(n - 1):
            sub_challenges[:, :, i] *= sub_challenges[:, :, i + 1]
        return sub_challenges

    @classmethod
    def efba_bit(cls, sub_challenges):
        """
        Converts sub-challenges to "efba" (extended for bias awareness) sub-challenges, i.e. appends a 1-bit to each
        sub-challenge.
        :param sub_challenges: A list of sub-challenge arrays, that is, an array of shape (N, k, n).
        :return: A list of "efba" sub-challenge arrays, that is, an array of shape (N, k, n+1).
        """
        return tools.append_last(sub_challenges, sub_challenges.dtype.type(1))

    @classmethod
    def normal_weights(cls, n, k, mu=0, sigma=1, random_instance=RandomState()):
        """
        Returns weights for an array of k LTFs of size n each.
        The weights are drawn from a normal distribution with given
        mean and std. deviation, if parameters are omitted, the
        standard normal distribution is used.
        The `normal` method of the optionally provided PRNG instance
        is used to obtain the weights. If no PRNG instance provided,
        a fresh `numpy.random.RandomState` instance is used.
        """
        return random_instance.normal(loc=mu, scale=sigma, size=(k, n))

    def __init__(self, weight_array, transform, combiner, bias=None):
        """
        Initializes an LTFArray based on given weight_array and
        combiner function with appropriate transformation of challenges.
        The bias is committed through the (n+1)th value in weight_array.
        So the parameter bias only states if the given weight_array has
        n+1 values (or not) while the challenges still has length n.
        :param bias: None, float or a two dimensional array of float with shape (k, 1)
                     This bias value or array of bias values will be appended to the weight_array.
                     Use a single value if you want the same bias for all weight_vectors.
        """
        (self.k, self.n) = shape(weight_array)
        self.weight_array = weight_array

        if isinstance(transform, CompoundTransformation):
            self.transform = transform.build()
        elif isinstance(transform, str):
            if not transform.startswith('transform_'):
                transform = 'transform_' + transform
            self.transform = getattr(self, transform)
        else:
            self.transform = transform

        if isinstance(combiner, str):
            if not combiner.startswith('combiner_'):
                combiner = 'combiner_' + combiner
            self.combiner = getattr(self, combiner)
        else:
            self.combiner = combiner

        # If necessary, convert bias definition
        if bias is None:
            self.bias = zeros(shape=(self.k, 1))
        elif isinstance(bias, float):
            self.bias = bias * ones(shape=(self.k, 1))
        elif isinstance(bias, ndarray) or isinstance(bias, list) and array(bias).shape == (self.k, ):
            self.bias = reshape(array(bias), (self.k, 1))
        else:
            self.bias = bias if isinstance(bias, ndarray) else array(bias)

        # Append bias values to weight array
        assert self.bias.shape == (self.k, 1),\
            'Expected bias to either have shape ({}, 1) or be a float, ' \
            'but got an array with shape {} and value {}.'.format(self.k, self.bias.shape, self.bias)
        self.weight_array = append(self.weight_array, self.bias, axis=1)

    def challenge_length(self) -> int:
        return self.weight_array.shape[1] - 1

    def response_length(self) -> int:
        return 1

    def eval(self, challenges, result_type=tools.BIT_TYPE):
        """
        Same es val, but only returns the sign of the responses.
        :param challenges: array of challenges of shape (N, n)
        :param result_type: numpy data type for result
        :return: array of responses of shape (N,)
        """
        return sign(self.val(challenges)).astype(result_type)

    def val(self, challenges):
        """
        Evaluates a given array of (master) challenges and returns the precise value of the combined LTFs responses.
        That is, the master challenges are first transformed into sub-challenges, using this LTFArray's transformation
        method. The challenges are then evaluated using ltf_eval. The responses are then combined using this LTFArray's
        combiner.
        :param challenges: array of shape(N,n)
                       Array of challenges which should be evaluated by the simulation.
        :return: array of float or int depending on the combiner of shape (N,)
                 Array of responses for the N different challenges.
        """
        return self.combiner(self.ltf_eval(self.transform(challenges, self.k)))

    def ltf_eval(self, sub_challenges):
        """
        This method evaluates a given array of sub-challenges.
        For this purpose it uses the dot product on every challenge and weight of the weight_array.
        :param sub_challenges: array of int shape(N,k,n)
                       Array of challenges which should be evaluated by the simulation.
        :return: array of int shape(N,k)
                 Array of responses for the N different challenges.
        """
        assert sub_challenges.shape[1:] == (self.k, self.n),\
            'Sub-challenges given to ltf_eval had shape {}, but shape (N, k, n) = (N, {}, {}) was expected.'.format(
                sub_challenges.shape, self.k, self.n
            )
        return self.core_eval(self.efba_bit(sub_challenges))

    def core_eval(self, efba_sub_challenges):
        """
        The core function that evaluates the LTFArray.
        :param efba_sub_challenges: (Extended for bias awareness sub challenges). Pre-processed challenges, i.e.
        sub-challenges that have an extra 1-bit at the end.
        Typically, this array is generated by processing a number of master-challenges with an input transformation
        into a list of sub-challenge arrays and then processing it with efba_bit.
        :return: The result of the LTFArray evaluation for each given array of "efba" sub-challenges
        """
        assert self.weight_array.shape == (self.k, self.n + 1), \
            'LTFArray\'s weight array was expected have shape (k, n+1) = {}, ' \
            'but had shape {} when core_eval was called.'.format((self.k, self.n + 1), self.weight_array.shape)
        if efba_sub_challenges.shape[2] == self.n + 1:
            return einsum('ji,...ji->...j', self.weight_array, efba_sub_challenges, optimize=True)
        elif efba_sub_challenges.shape[2] == self.n:
            return einsum('ji,...ji->...j', self.weight_array[:, :-1], efba_sub_challenges, optimize=True)
        else:
            raise ValueError(f'Challenges given to LTFArray.core_eval must be of shape (N, k, n) for bias-unaware '
                             f'evaluation, and of shape (N, k, n+1) for bias-aware evaluation. This LTFArray has '
                             f'k={self.k} and n={self.n}, but challenges given had shape {efba_sub_challenges.shape}.')


class NoisyLTFArray(LTFArray):
    """
    Class that simulates k LTFs with n bits and a constant term each
    with noise effect and constant bias added.
    """

    @staticmethod
    def sigma_noise_from_random_weights(n, sigma_weight, noisiness=0.1):
        """
        returns sd of noise (sigma_noise) out of n stages with
        sd of weight differences (sigma_weight) and noisiness factor
        """
        return sqrt(n) * sigma_weight * noisiness

    @staticmethod
    def init_normal_empirical(n, k, transform, combiner, intra_dist, random_instance=RandomState(), bias=None,
                              approx_threshold=.1):
        """
        Initializes a NoisyLTFArray with given parameters that can be expected to have the given intra_dist.
        :param n: length of challenges
        :param k: number of LTFs in the array
        :param transform: input transformation for the LTF array
        :param combiner: function mapping the individual output bits to one output bit
        :param intra_dist: desired intra_dist, defined as the probability to see same output on two evaluations using
                           the same challenge.
        :param random_instance: pseudorandom generator to be used
        :param bias: bias of the LTF array
        :return: NoisyLTFArray
        """
        assert intra_dist > 0

        instance = NoisyLTFArray(
            weight_array=LTFArray.normal_weights(n, k, random_instance=random_instance),
            transform=transform,
            combiner=combiner,
            sigma_noise=1,
            random_instance=random_instance,
            bias=bias,
        )

        # double max_sigma_noise until large enough
        while tools.approx_dist(instance, instance, 1000) < intra_dist:
            instance.sigma_noise *= 2
        min_sigma_noise = 0
        max_sigma_noise = 2*instance.sigma_noise

        # binary search in [0, max_sigma_noise]
        instance.sigma_noise = (max_sigma_noise + min_sigma_noise) / 2
        estimation_distance = tools.approx_dist(instance, instance, 10000, random_instance)
        while abs(intra_dist - estimation_distance) > approx_threshold:

            # update interval bounds
            if estimation_distance > intra_dist:
                max_sigma_noise = instance.sigma_noise
            elif estimation_distance <= intra_dist:
                min_sigma_noise = instance.sigma_noise

            # update instance and estimated distance
            instance.sigma_noise = (max_sigma_noise + min_sigma_noise) / 2
            estimation_distance = tools.approx_dist(instance, instance, 10000, random_instance)

        return instance

    def __init__(self, weight_array, transform, combiner, sigma_noise,
                 random_instance=RandomState(), bias=None):
        """
        Initializes LTF array like in LTFArray and uses the provided
        PRNG instance for drawing noise values. If no PRNG provided, a
        fresh `numpy.random.RandomState` instance is used.
        :param bias: None, float or a two dimensional array of float with shape (k, 1)
                     This bias value or array of bias values will be appended to the weight_array.
                     Use a single value if you want the same bias for all weight_vectors.
        """
        super().__init__(weight_array, transform, combiner, bias)
        self.sigma_noise = sigma_noise
        self.random = random_instance

    def ltf_eval(self, sub_challenges):
        """
        Calculates weight_array with given set of challenges including noise.
        The noise effect is a normal distributed random variable with mu=0,
        sigma=sigma_noise.
        Random numbers are drawn from the PRNG instance generated when
        initializing the NoisyLTFArray.
        """
        evaled_inputs = super().ltf_eval(sub_challenges)
        noise = self.random.normal(loc=0, scale=self.sigma_noise, size=(len(evaled_inputs), self.k))
        return evaled_inputs + noise


class SimulationMajorityLTFArray(LTFArray):
    """
    This class provides a majority vote version of the NoisyLTFArray.
    It uses different noises for each PUF instance and each challenge input.
    Majority vote means that each fo the k PUFs get evaluated vote_count times
    in order to mitigate the impact of noise to the responses. With this class
    it is possible to simulate quite stable huge PUF systems.
    This class can be used as PUF simulation in order to generate a trainingset.
    """

    def __init__(self, weight_array, transform, combiner, sigma_noise,
                 random_instance_noise=RandomState(), bias=None, vote_count=1):
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
        :param random_instance_noise: RandomState
                                      This pseudo-random number generator is used to generate noise.
        :param bias: None, float or a two dimensional array of float with shape (k, 1)
                     This bias value or array of bias values will be appended to the weight_array.
                     Use a single value if you want the same bias for all weight_vectors.
        :param vote_count: positive odd int
                           Number which defines the number of evaluations of PUFs in oder to majority vote the output.
        """
        super().__init__(weight_array, transform, combiner, bias=bias)
        self.sigma_noise = sigma_noise
        self.random = random_instance_noise
        # majority vote only works with an odd number of votes
        assert vote_count % 2 == 1
        self.vote_count = vote_count

    def val(self, challenges):
        """
        This function a calculates the output of the LTFArray based on weights with majority vote.
        :param challenges: array of int shape(N,k,n)
                       Array of challenges which should be evaluated by the simulation.
        :return: array of int shape(N)
                 Array of responses for the N different challenges.
        """
        return self.combiner(self.majority_vote(self.transform(challenges, self.k)))

    def majority_vote(self, sub_challenges):
        """
        This function evaluates transformed input challenges and uses majority vote on them.
        :param sub_challenges: array of int with shape(N,k,n)
                                   Array of transformed input challenges.
        :return: array of int with shape(N,k,n)
                 Majority voted responses for each of the k PUFs.
        """
        # Evaluate the sub challenges individually
        (N, k, _) = sub_challenges.shape
        evaluated_sub_challenges = super().ltf_eval(sub_challenges)

        # Duplicate the evaluation result for each vote and add individual noise
        # Note the votes are on the first axis
        evaled_inputs = broadcast_to(evaluated_sub_challenges, (self.vote_count, N, k))
        noise = self.random.normal(scale=self.sigma_noise, size=(self.vote_count, N, k))

        # Majority vote (i.e., sign(sum(·))) along the first axis
        return sign(np_sum(sign(evaled_inputs + noise), axis=0))
