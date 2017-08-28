from numpy import prod, shape, sign, dot, array, tile, transpose, concatenate, dstack, swapaxes, sqrt, amax, vectorize
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
    def transform_lightweight_secure_original(cs, k):
        """
        Input transform as defined by Majzoobi et al. 2008.
        """
        N = len(cs)
        n = len(cs[0])
        assert n % 2 == 0, 'Secure Lightweight Input Transformation only defined for even n. Sorry!'

        cs_shift_trans = __class__.transform_shift_lightweight_secure(cs, k)

        """ Perform atf transform """
        result = transpose(
            array([
                prod(cs_shift_trans[:, :, i:], 2)
                for i in range(n)
            ]),
            (1, 2, 0)
        )

        assert result.shape == (N, k, n), 'The resulting challenges have not the desired shape. Sorry!'

        return result

    @staticmethod
    def transform_lightweight_secure(cs, k):
        """
        Input transform as defined by Majzoobi et al. 2008, but with the shift
        operation executed after and without ATF transform.
        """
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
    def transform_shift_lightweight_secure(cs, k):
        """
        Input transform as defined by Majzoobi et al. 2008, with the shift
        operation executed first and without ATF transform.
        """
        N = len(cs)
        n = len(cs[0])
        assert n % 2 == 0, 'Secure Lightweight Input Transformation only defined for even n. Sorry!'

        shifted = __class__.transform_shift(cs, k)

        cs = transpose(
            concatenate(
                (
                    [ shifted[:,:,i] * shifted[:,:,i+1] for i in range(0, n, 2) ],
                    [ shifted[:,:,0] ],
                    [ shifted[:,:,i] * shifted[:,:,i+1] for i in range(1, n-2, 2) ],
                )
            ),
            (1, 2, 0)
        )

        assert cs.shape == (N, k, n)

        return cs

    @staticmethod
    def transform_soelter_lightweight_secure(cs, k):
        """
        Input transformation like defined by Majzoobi et al. (cf. transform_lightweight_secure),
        but differs in one bit. Introduced by Sölter.
        """
        N = len(cs)
        n = len(cs[0])
        assert n % 2 == 0, 'Sölter\'s Secure Lightweight Input Transformation only defined for even n. Sorry!'
        n_half = int(n/2)

        cs = transpose(
            concatenate(
                (
                    [cs[:, i] * cs[:, i + 1] for i in range(0, n, 2)],      # ( x1x2, x3x4, ... xn-1xn )
                    [cs[:, n_half]],                                        # ( x_(n/2+1) )
                    [cs[:, i] * cs[:, i + 1] for i in range(1, n - 2, 2)],  # ( x2x3, x4x5, ... xn-2xn-1 )
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
    def transform_polynomial(cs, k):
        """
        This input transformation interprets a challenge c as a
        polynomial over the finite field GF(2^n)=F2/f*F2, where f is a
        irreducible polynomial of degree n.
        The irreducible polynomial f is hard coded and
        of degree 8, 16, 24, 32, 48, or 64.
        Each Arbiter Chain i receives as input the polynomial c^i
        as element of GF(2^n).
        """

        N = len(cs)
        n = len(cs[0])
        assert n in [8, 16, 24, 32, 48, 64], 'Polynomial transformation is only implemented for challenges with n in {8, 16, 24, 32, 48, 64}. ' \
                                       'Sorry!'
        if n == 64:
            f = [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 1]
        elif n == 48:
            f = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1,
                 0, 0, 1]
        elif n == 32:
            f = [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        elif n == 24:
            f = [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        elif n == 16:
            f = [1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1]
        elif n == 8:
            f = [1, 0, 1, 0, 0, 1, 1, 0, 1]

        """ Transform challenge to 0,1 array to compute transformation with numpy. """
        vtransform_to_01 = vectorize(tools.transform_challenge_11_to_01)
        cs_01 = array([vtransform_to_01(c) for c in cs])

        """ Compute c^i for each challenge for i from 1 to k. """
        cs = concatenate([
                [tools.poly_mult_div(c, f, k) for c in cs_01]
        ])

        """ Transform challenges back to -1,1 notation. """
        vtransform_to_11 = vectorize(tools.transform_challenge_01_to_11)
        result = array([vtransform_to_11(c) for c in cs])

        assert result.shape == (N, k, n), 'The resulting challenges have not the desired shape. Sorry!'
        return result

    @staticmethod
    def transform_permutation_atf(cs, k):
        """
        This transformation performs first a pseudorandom permutation of the challenge k times before applying the
        ATF transformation to each challenge.
        :param cs:
        :param k:
        :return:
        """
        N = len(cs)
        n = len(cs[0])
        seed = 0x1234

        """ Perform random permutations """
        cs_permuted = array(
            [
                [RandomState(seed + i).permutation(c)
                 for i in range(k)]
                 for c in cs
                ]
        )
        """ Perform atf transform """
        result = transpose(
            array([
                      prod(cs_permuted[:, :, i:], 2)
                      for i in range(n)
                      ]),
            (1, 2, 0)
        )

        assert result.shape == (N, k, n), 'The resulting challenges have not the desired shape. Sorry!'

        return result

    @staticmethod
    def transform_random(cs, k):
        """
        This input transformation chooses for each Arbiter Chain an random challenge based on the initial challenge.
        """

        N = len(cs)
        n = len(cs[0])

        vtransform_to_01 = vectorize(tools.transform_challenge_11_to_01)
        cs_01 = array([vtransform_to_01(c) for c in cs])

        result = array([RandomState(c).choice((-1, 1), (k, n)) for c in cs_01])

        assert result.shape == (N, k, n), 'The resulting challenges have not the desired shape. Sorry!'
        return result

    @staticmethod
    def transform_stack(transform_1, kk, transform_2):
        """
        Returns an input transformation that will transform the first kk challenges using transform_1,
        the remaining k - kk challenges using transform_2.
        :return: A function that can perform the desired transformation
        """
        def transform(cs, k):
            (N,n) = cs.shape
            transformed_1 = transform_1(cs, kk)
            transformed_2 = transform_2(cs, k - kk)
            assert transformed_1.shape == (N, kk, n)
            assert transformed_2.shape == (N, k - kk, n)
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
                                 kk,
                                 transform_2.__name__.replace('transform_', '')
                             )

        return transform

    @staticmethod
    def transform_permutations(seed, nn, kk, atf=False):
        """
        Returns an input transformation that uses k pseudorandomly generated permutations
        :param seed: Seed for the pseudorandom generation
        :param nn: challenge length (must equal LTFArray.n)
        :param kk: Number of permutations to be used (must equal LTFArray.k)
        :param atf: Perform ATF transform after permuting
        :return: The desired input transform
        """
        r = RandomState(seed)
        permutations = [r.permutation(nn) for x in range(kk)]

        def transform(cs, k):
            (N, n) = cs.shape
            assert k == kk and n == nn, \
                'Permutations Input Transform cannot be used for LTFArrays with size other than defined'

            result = swapaxes(
                array([
                    cs[:, permutations[i]]
                    for i in range(kk)
                ]),
                0,
                1
            )

            if atf:
                """ Perform atf transform """
                result = transpose(
                    array([
                        prod(result[:, :, i:], 2)
                        for i in range(n)
                    ]),
                    (1, 2, 0)
                )

            return result

        transform.__name__ = 'transform_permutations' + ('_plus_atf_' if atf else '') + '_%x' % seed
        return transform

    @staticmethod
    def transform_concat(transform_1, nn, transform_2):
        """
        Returns an input transformation that will transform the first nn bit of each challenge using transform_1,
        the remaining bits using transform_2.
        :return: A function that can perform the desired transformation
        """
        def transform(cs, k):
            (N,n) = cs.shape
            cs1 = cs[:,:nn]
            cs2 = cs[:,nn:]
            transformed_1 = transform_1(cs1, k)
            transformed_2 = transform_2(cs2, k)
            assert transformed_1.shape == (N, k, nn)
            assert transformed_2.shape == (N, k, n - nn)
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
                                 nn,
                                 transform_2.__name__.replace('transform_', '')
                             )

        return transform

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
        :param inputs: list of challenges
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
        evaled_inputs = super().ltf_eval(inputs)
        noise = self.random.normal(loc=0, scale=self.sigma_noise, size=(len(evaled_inputs), self.k))
        return evaled_inputs + noise
