from numpy import prod, shape, random, sign, dot, concatenate, array, full, tile, transpose, concatenate, dstack, swapaxes


class LTFArray():
    """
    Class that simulates k LTFs with n bits and a constant term each.
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
        Input transformation that simulations an Arbiter PUF
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
        assert n == 64, 'MM transform currently only implemented for n=64. Sorry!'

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
    def normal_weights(n, k, mu=0, sigma=1):
        """
        Returns weights for an array of k LTFs of size n each.
        The weights are drawn from a normal distribution with given
        mean and std. deviation, if parameters are omitted, the
        standard normal distribution is used.
        """
        return random.normal(loc=mu, scale=sigma, size=(k, n))

    def __init__(self, weight_array, transform, combiner):
        (self.k, self.n) = shape(weight_array)
        self.weight_array = weight_array
        self.transform = transform
        self.combiner = combiner

    def eval(self, inputs):
        """
        evaluates a given list of challenges
        :param x: list of challenges
        :return: list of responses
        """
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
