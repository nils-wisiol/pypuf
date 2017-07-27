from numpy import prod, shape, sign, dot, array, tile, transpose, concatenate, dstack, swapaxes, sqrt, amax
from numpy.random import RandomState
from pypuf import tools
from pypuf.simulation.base import Simulation


class LTFArray(Simulation):
    """
    Class that simulates k LTFs with n bits and a constant term each
    and constant bias added.
    """

    @staticmethod
    def combiner_xor(r):
        """
        combines output responses with the XOR operation
        :param r: a list with a number of vectors of single LTF results
        :return: a list of full results, one for each
        """
        return prod(r, 1)

    @staticmethod
    def combiner_ip_mod2(r):
        """
        combines output responses with the inner product mod 2 operation
        :param r: a list with a number of vectors of single LTF results
        :return: a list of full results, one for each
        """
        n = len(r[0])
        assert n % 2 == 0, 'IP mod 2 is only defined for even n. Sorry!'
        return prod(
            transpose(
                [
                    amax((r[:,i], r[:,i+1]), 0)
                    for i in range(0, n, 2)
                ])
            , 1)

    @staticmethod
    def transform_id(cs, k):
        """
        Input transformation that does nothing.
        :return:
        """
        return array([
                tile(c, (k, 1))  # same unmodified challenge for all k LTFs
                for c in cs
            ])

    @staticmethod
    def transform_atf(cs, k):
        """
        Input transformation that simulates an Arbiter PUF
        :return:
        """

        # Transform with ATF monomials
        cs = transpose(
            array([
                prod(cs[:,i:], 1)
                for i in range(len(cs[0]))
            ])
        )

        # Same challenge for all k Arbiters
        return __class__.transform_id(cs, k)

    @staticmethod
    def transform_mm(cs, k):
        N = len(cs)
        n = len(cs[0])
        assert k == 2, 'MM transform currently only implemented for k=2. Sorry!'
        assert n % 2 == 0, 'MM transform only defined for even n. Sorry!'

        cs_1 = cs
        cs_2 = transpose(
                concatenate(
                (
                    [ cs[:,0] ],
                    [ cs[:,i] * cs[:,i+1] for i in range(0, n, 2) ],
                    [ cs[:,i] * cs[:,i+1] * cs[:,i+2] for i in range(0, n-2, 2) ]
                )
            )
        )

        result = swapaxes(dstack((cs_1, cs_2)), 1, 2)
        assert result.shape == (N, 2, n)
        return result

    @staticmethod
    def transform_lightweight_secure(cs, k):
        N = len(cs)
        n = len(cs[0])
        assert n % 2 == 0, 'Secure Lightweight Input Transformation only defined for even n. Sorry!'

        cs = transpose(
                concatenate(
                (
                    [ cs[:,i] * cs[:,i+1] for i in range(0, n, 2) ],    # ( x1x2, x3x4, ... xn-1xn )
                    [ cs[:,0] ],                                        # ( x1 )
                    [ cs[:,i] * cs[:,i+1] for i in range(1, n-2, 2) ],  # ( x2x3, x4x5, ... xn-2xn-1 )
                )
            )
        )

        assert cs.shape == (N, n)

        return __class__.transform_shift(cs, k)

    @staticmethod
    def transform_shift(cs, k):

        N = len(cs)
        n = len(cs[0])

        result = swapaxes(array([
            concatenate((cs[:,l:], cs[:,:l]), axis=1)
            for l in range(k)
        ]), 0, 1)

        assert result.shape == (N, k, n)

        return result

    @staticmethod
    def transform_1_n_bent(cs, k):
        """
        For one LTF, we compute the input as follows: the i-th input bit will be the result
        of the challenge shifted by i bits to the left, then input into inner product mod 2
        function.
        The other LTF get the original input.
        """
        N = len(cs)
        n = len(cs[0])
        assert k >= 2, '1-n bent transform currently only implemented for k>=2. Sorry!'
        assert n % 2 == 0, '1-n bent transform only defined for even n. Sorry!'

        shift_challenges = __class__.transform_shift(cs, n)
        assert shift_challenges.shape == (N, n, n)

        bent_challenges = transpose(
            array(
                [
                    __class__.combiner_ip_mod2(shift_challenges[:,i,:])
                    for i in range(n)
                ]
            )
        )
        assert bent_challenges.shape == (N, n)

        return array([
                concatenate(
                    (
                        [bent_challenges[j]],    # 'bent' challenge as generated above
                        tile(cs[j], (k - 1, 1))  # unmodified challenge for k-1 LTFs
                    ),
                    axis=0
                )
                for j in range(N)
            ])

    @staticmethod
    def transform_1_1_bent(cs, k):
        """
        For one LTF, we compute the input as follows: the first input bit will be
        the result of IPmod2 of the original challenge, all other input bits will
        remain the same.
        The other LTF get the original input.
        """
        N = len(cs)
        n = len(cs[0])
        assert k >= 2, '1-n bent transform currently only implemented for k>=2. Sorry!'
        assert n % 2 == 0, '1-n bent transform only defined for even n. Sorry!'

        bent_challenge_bits = __class__.combiner_ip_mod2(cs)
        assert bent_challenge_bits.shape == (N, )

        return array([
                concatenate(
                    (
                        [concatenate(([[bent_challenge_bits[j]], cs[j][1:]]))],  # 'bent' challenge bit plus remainder unchanged
                        tile(cs[j], (k - 1, 1))                  # unmodified challenge for k-1 LTFs
                    ),
                    axis=0
                )
                for j in range(N)
            ])

    @staticmethod
    def normal_weights(n, k, mu=0, sigma=1, random_instance=RandomState()):
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

    def __init__(self, weight_array, transform, combiner, bias=False):
        """
        Initializes an LTFArray based on given weight_array and
        combiner function with appropriate transformation of challenges.
        The bias is committed through the (n+1)th value in weight_array.
        So the parameter bias only states if the given weight_array has
        n+1 values (or not) while the challenges still has length n.
        """
        (self.k, self.n) = shape(weight_array)
        self.weight_array = weight_array
        self.transform = transform
        self.combiner = combiner
        self.bias = bias

    def eval(self, inputs):
        """
        evaluates a given list of challenges regarding bias
        :param x: list of challenges
        :return: list of responses
        """
        if self.bias:
            inputs = tools.iter_append_last(inputs, 1)
        return sign(self.val(inputs))

    def val(self, inputs):
        return self.combiner(self.ltf_eval(self.transform(inputs, self.k)))

    def ltf_eval(self, inputs):
        """
        :return: array
        """
        return transpose(
            array([
                dot(
                    inputs[:,l],
                    self.weight_array[l]
                )
                for l in range(self.k)
            ])
        )


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

    def __init__(self, weight_array, transform, combiner, sigma_noise,
                 random_instance=RandomState(), bias=False):
        """
        Initializes LTF array like in LTFArray and uses the provided
        PRNG instance for drawing noise values. If no PRNG provided, a
        fresh `numpy.random.RandomState` instance is used.
        """
        super().__init__(weight_array, transform, combiner, bias)
        self.sigma_noise = sigma_noise
        self.random = random_instance

    def ltf_eval(self, inputs):
        """
        Calculates weight_array with given set of challenges including noise.
        The noise effect is a normal distributed random variable with mu=0,
        sigma=sigma_noise.
        Random numbers are drawn from the PRNG instance generated when
        initializing the NoisyLTFArray.
        """
        noise = self.random.normal(loc=0, scale=self.sigma_noise, size=(1, self.k))
        return super().ltf_eval(inputs) + noise
