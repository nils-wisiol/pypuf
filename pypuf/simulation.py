from numpy import prod, shape, random, sign, dot, array, tile, transpose, concatenate, dstack, swapaxes, sqrt, append
from pypuf import tools

class LTFArray():
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
    def normal_weights(n, k, mu=0, sigma=1):
        """
        Returns weights for an array of k LTFs of size n each.
        The weights are drawn from a normal distribution with given
        mean and std. deviation, if parameters are omitted, the
        standard normal distribution is used.
        """
        return random.normal(loc=mu, scale=sigma, size=(k, n))

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
                 noise_seed=None, bias=False):
        """
        Initializes LTF array like in LTFArray with the sd of a random
        noise effect added.
        """
        super().__init__(weight_array, transform, combiner, bias)
        self.sigma_noise = sigma_noise
        if noise_seed is not None:
            random.seed(noise_seed)

    def ltf_eval(self, inputs):
        """
        Calculates weight_array with given set of challenges including noise.
        The noise effect is a normal distributed random variable with mu=0,
        sigma=sigma_noise.
        It can be seeded optionally by an integer out of [0, 2^32 -1].
        """
        noise = random.normal(loc=0, scale=self.sigma_noise, size=(1,self.k))
        return super().ltf_eval(self, inputs) + noise
