"""
This module provides several different implementations of arbiter PUF simulations. The linear threshold function array
model is the core of each simulation class.
"""
from numpy import sum as np_sum, ones, ndarray, zeros, reshape, broadcast_to, einsum
from numpy import prod, shape, sign, array, transpose, concatenate, swapaxes, sqrt, amax, append, int8
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
        :param challenges: array of int8 shape(N,n)
                           Array of challenges which should be evaluated by the simulation.
        :param k: int
                  Number of LTFArray PUFs
        :return:  array of int8 shape(N,k,n)
                  Array of transformed challenges.
        """
        (N, n) = challenges.shape
        return transpose(broadcast_to(challenges, (k, N, n)), axes=(1, 0, 2))

    @classmethod
    def transform_atf(cls, challenges, k):
        """
        Input transformation that simulates an Arbiter PUF
        :param challenges: array of int8 shape(N,n)
                           Array of challenges which should be evaluated by the simulation.
        :param k: int
                  Number of LTFArray PUFs
        :return:  array of int shape(N,k,n)
                  Array of transformed challenges.
        """
        tools.assert_result_type(challenges)
        # Transform with ATF monomials
        challenges = transpose(
            array([
                prod(challenges[:, i:], 1, dtype=tools.BIT_TYPE)
                for i in range(len(challenges[0]))
            ], dtype=tools.BIT_TYPE)
        )

        # Same challenge for all k Arbiters
        res = cls.transform_id(challenges, k)
        tools.assert_result_type(res)
        return res

    @classmethod
    def transform_lightweight_secure(cls, challenges, k):
        """
        Input transform as defined by Majzoobi et al. 2008.
        :param challenges: array of int8 shape(N,n)
                           Array of challenges which should be evaluated by the simulation.
        :param k: int
                  Number of LTFArray PUFs
        :return:  array of int8 shape(N,k,n)
                  Array of transformed challenges.
        """
        tools.assert_result_type(challenges)
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
        tools.assert_result_type(result)
        return result

    @classmethod
    def transform_soelter_lightweight_secure(cls, challenges, k):
        """
        Input transformation like defined by Majzoobi et al. (cf. transform_lightweight_secure),
        but differs in one bit. Introduced by Sölter.
        :param challenges: array of int8 shape(N,n)
                           Array of challenges which should be evaluated by the simulation.
        :param k: int
                  Number of LTFArray PUFs
        :return:  array of int8 shape(N,k,n)
                  Array of transformed challenges.
        """
        tools.assert_result_type(challenges)
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
        tools.assert_result_type(res)
        return res

    @classmethod
    def transform_shift(cls, challenges, k):
        """
        Input transformation that shifts the input bits for each of the k PUFs.
        :param challenges: array of int8 shape(N,n)
                           Array of challenges which should be evaluated by the simulation.
        :param k: int
                  Number of LTFArray PUFs
        :return:  array of int8 shape(N,k,n)
                  Array of transformed challenges.
        """
        tools.assert_result_type(challenges)
        N = len(challenges)
        n = len(challenges[0])

        result = swapaxes(array([
            concatenate((challenges[:, l:], challenges[:, :l]), axis=1)
            for l in range(k)
        ]), 0, 1)

        assert result.shape == (N, k, n)
        tools.assert_result_type(result)
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
        :param challenges: array of int8 shape(N,n)
                           Array of challenges which should be evaluated by the simulation.
        :param k: int
                  Number of LTFArray PUFs
        :return:  array of int8 shape(N,k,n)
                  Array of transformed challenges.
        """
        tools.assert_result_type(challenges)
        N = len(challenges)
        n = len(challenges[0])
        assert n in [8, 16, 24, 32, 48, 64], 'Polynomial transformation is only implemented for challenges with n in ' \
                                             '{8, 16, 24, 32, 48, 64}.'
        if n == 64:
            irreducible_polynomial = array(
                [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                dtype=tools.BIT_TYPE
            )
        elif n == 48:
            irreducible_polynomial = array(
                [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1], dtype=tools.BIT_TYPE
            )
        elif n == 32:
            irreducible_polynomial = array(
                [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                dtype=tools.BIT_TYPE
            )
        elif n == 24:
            irreducible_polynomial = array([1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                           dtype=tools.BIT_TYPE)
        elif n == 16:
            irreducible_polynomial = array([1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1], dtype=tools.BIT_TYPE)
        elif n == 8:
            irreducible_polynomial = array([1, 0, 1, 0, 0, 1, 1, 0, 1], dtype=tools.BIT_TYPE)
        tools.assert_result_type(irreducible_polynomial)

        # Transform challenge to 0,1 array to compute transformation with numpy.
        cs_01 = array([tools.transform_challenge_11_to_01(c) for c in challenges])

        # Compute c^i for each challenge for i from 1 to k.
        challenges = concatenate([
            [tools.poly_mult_div(c, irreducible_polynomial, k) for c in cs_01]
        ])
        tools.assert_result_type(challenges)

        # Transform challenges back to -1,1 notation.
        result = array([tools.transform_challenge_01_to_11(c) for c in challenges], dtype=tools.BIT_TYPE)

        assert result.shape == (N, k, n), 'The resulting challenges have not the desired shape.'
        tools.assert_result_type(result)
        return result

    @classmethod
    def transform_permutation_atf(cls, challenges, k):
        """
        This transformation performs first a pseudorandom permutation of the challenge k times before applying the
        ATF transformation to each challenge.
        :param challenges: array of int8 shape(N,n)
                           Array of challenges which should be evaluated by the simulation.
        :param k: int
                  Number of LTFArray PUFs
        :return:  array of int8 shape(N,k,n)
                  Array of transformed challenges.
        """
        tools.assert_result_type(challenges)
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
        tools.assert_result_type(result)
        return result

    @classmethod
    def transform_random(cls, challenges, k):
        """
        This input transformation chooses for each Arbiter Chain an random challenge based on the initial challenge.
        :param challenges: array of int8 shape(N,n)
                           Array of challenges which should be evaluated by the simulation.
        :param k: int
                  Number of LTFArray PUFs
        :return:  array of int8 shape(N,k,n)
                  Array of transformed challenges.
        """
        tools.assert_result_type(challenges)
        N = len(challenges)
        n = len(challenges[0])

        cs_01 = array([tools.transform_challenge_11_to_01(c) for c in challenges], dtype=tools.BIT_TYPE)

        result = array([RandomState(c).choice((-1, 1), (k, n)) for c in cs_01], dtype=tools.BIT_TYPE)

        assert result.shape == (N, k, n), 'The resulting challenges have not the desired shape.'
        tools.assert_result_type(result)
        return result

    @classmethod
    def transform_fixed_permutation(cls, challenges, k):
        """
        Permutes the challenge bits using hardcoded, fix point free permutations designed such that no
        sub-challenge bit gets permuted equally for all other generated sub-challenges. Such permutations
        are not easy to find, hence this function only supports a limited number of n and k.
        After permutation, we apply the ATF transform to the sub-challenges (hence generating what in the
        linear Arbiter PUF model is called feature vectors).
        :param challenges: array of int8 shape(N,n)
                           Array of challenges which should be evaluated by the simulation.
        :param k: int
                  Number of LTFArray PUFs
        :return:  array of int8 shape(N,k,n)
                  Array of transformed challenges.
        """
        FIXED_PERMUTATION_SEEDS = {
            64: [2989, 2992, 3038, 3084, 3457, 6200, 7089, 18369, 21540, 44106],
        }

        # check parameter n
        n = len(challenges[0])
        assert n in FIXED_PERMUTATION_SEEDS.keys(), 'Fixed permutation currently not supported for n=%i, but only ' \
                                                    'for n in %s' % (n, FIXED_PERMUTATION_SEEDS.keys())

        # check parameter k
        seeds = FIXED_PERMUTATION_SEEDS[n]
        assert k <= len(seeds), 'Fixed permutation for n=%i currently only supports k<=%i.' % (n, len(seeds))

        # generate permutations
        permutations = [RandomState(seed).permutation(n) for seed in seeds]

        # perform permutations
        result = swapaxes(
            array([
                challenges[:, permutations[i]]
                for i in range(k)
            ], dtype=int8),
            0,
            1
        )

        result = cls.att(result)

        return result

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
           :param challenges: array of int8 shape(N,n)
                              Array of challenges which should be evaluated by the simulation.
           :param k: int
                     Number of LTFArray PUFs
           :return: A function: array of int with shape(N,n), int number of PUFs k -> shape(N,k,n)
                    A function that can perform the desired transformation.
           """
            tools.assert_result_type(challenges)
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
            :param challenges: array of int8 shape(N,n)
                               Array of challenges which should be evaluated by the simulation.
            :param k: int
                     Number of LTFArray PUFs
            :return: A function: array of int with shape(N,n), int number of PUFs k -> shape(N,k,n)
                     A function that can perform the desired transformation.
            """
            tools.assert_result_type(challenges)
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
            :param challenges: array of int8 shape(N,n)
                               Array of challenges which should be evaluated by the simulation.
            :param k: int
                      Number of LTFArray PUFs
            :return: A function: array of int with shape(N,n), int number of PUFs k -> shape(N,k,n)
                     A function that can perform the desired transformation.
            """
            tools.assert_result_type(challenges)
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
        return tools.append_last(sub_challenges, tools.BIT_TYPE(1))

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

    def eval(self, challenges):
        """
        Same es val, but only returns the sign of the responses.
        :param challenges: array of int8 challenges of shape (N, n)
        :return: array of int8 responses of shape (N,)
        """
        return sign(self.val(challenges)).astype(tools.BIT_TYPE)

    def val(self, challenges):
        """
        Evaluates a given array of (master) challenges and returns the precise value of the combined LTFs responses.
        That is, the master challenges are first transformed into sub-challenges, using this LTFArray's transformation
        method. The challenges are then evaluated using ltf_eval. The responses are then combined using this LTFArray's
        combiner.
        :param challenges: array of int8 shape(N,n)
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
        assert efba_sub_challenges.dtype == tools.BIT_TYPE,\
            'LTFArrays can only be evaluated on challenges of type {}, but got type {}.'.format(
                tools.BIT_TYPE, efba_sub_challenges.dtype
            )
        assert efba_sub_challenges.shape[1:] == (self.k, self.n + 1),\
            '"Efba" sub-challenges given to core_eval must be of shape (N, k, n+1). Expected shape was (N, {}, {}),' \
            'but got {}.\n' \
            'Remember that all sub-challenges given to core_eval must have the "efba" input bit, i.e. the last bit' \
            'of all sub-challenges must be 1. Hence, the third dimension must have length n+1.'.format(
                self.k, self.n + 1, efba_sub_challenges.shape
            )
        assert self.weight_array.shape == (self.k, self.n + 1),\
            'LTFArray\'s weight array was expected have shape (k, n+1) = {}, '\
            'but had shape {} when core_eval was called.'.format((self.k, self.n + 1), self.weight_array.shape)
        return einsum('ji,...ji->...j', self.weight_array, efba_sub_challenges, optimize=True)


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
        # Evaluate the sub challeneges individually
        (N, k, _) = sub_challenges.shape
        evaluated_sub_challenges = super().ltf_eval(sub_challenges)

        # Duplicate the evaluation result for each vote and add individual noise
        # Note the votes are on the first axis
        evaled_inputs = broadcast_to(evaluated_sub_challenges, (self.vote_count, N, k))
        noise = self.random.normal(scale=self.sigma_noise, size=(self.vote_count, N, k))

        # Majority vote (i.e., sign(sum(·))) along the first axis
        return sign(np_sum(sign(evaled_inputs + noise), axis=0))
