from numpy import sign, dot, around, exp, array, seterr, minimum, abs, full, count_nonzero, amin, amax, double
from numpy.random import RandomState
from numpy.linalg import norm
from pypuf.learner.base import Learner
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.tools import compare_functions


class LogisticRegression(Learner):
    """
    Learn an LTF Array with Logistic Regression.

    This class provides a Logistic Regression learner with automatically generated
    models that fit the LTF Array as defined in the constructor.
    """

    class ModelUpdate(object):
        """
        Model update according to the naive algorithm. Works, but is really slow to converge.
        """

        def __init__(self, model):
            model = model

        def update(self, gradient):
            """
            Use the gradient scaled with a constant to determine the update step.
            """
            return -.3 * gradient

    class RPropModelUpdate(ModelUpdate):
        """
        Model update according to the Resilient Backpropagation algorithm. For details, see update() method.
        """

        def __init__(self, model, eta_minus=0.5, eta_plus=1.2):

            self.n = n = model.n
            self.k = k = model.k

            self.eta_minus = eta_minus
            self.eta_plus = eta_plus
            self.delta_min = 10**-4
            self.delta_max = 10**+1
            self.last_gradient = full((k, n), 1.0)
            self.last_step_size = full((k, n), 0.0)
            self.step_size = full((k, n), 1.0)
            self.step = full((k, n), 0.0)
            self.step_size_max = full(self.n, self.delta_max, double)
            self.step_size_min = full(self.n, self.delta_min, double)

            super().__init__(model)

        def update(self, grad):
            """
            Compute update step according to "Resilient Backpropagation" by
            Riedmiller, Martin, and Heinrich Braun. "A direct adaptive method for faster backpropagation learning: The RPROP algorithm."
            Neural Networks, 1993., IEEE International Conference on. IEEE, 1993.

            Implementation following the neat implementation used in
            Rührmair, Ulrich, et al. "Modeling attacks on physical unclonable functions."
            Proceedings of the 17th ACM conference on Computer and communications security.
            ACM, 2010.

            For their original code, please see http://www.pcp.in.tum.de/code/lr.zip,
            predictor.py:299
            """

            for l, grad_l in enumerate(grad):
                step_indicator = sign(grad_l * self.last_gradient[l])

                self.step_size[l][step_indicator > 0] *= self.eta_plus
                self.step_size[l][step_indicator < 0] *= self.eta_minus

                self.step_size[l] = amin((self.step_size[l], self.step_size_max), 0)
                self.step_size[l] = amax((self.step_size[l], self.step_size_min), 0)

                self.step[l][step_indicator > 0] = -(self.step_size[l][step_indicator > 0] * sign(grad_l[step_indicator > 0]))
                self.step[l][step_indicator < 0] = -self.last_step_size[l][step_indicator < 0]
                self.step[l][step_indicator == 0] = -self.step_size[l][step_indicator == 0] * sign(grad_l[step_indicator == 0])

                self.last_gradient[l] = grad_l
                self.last_gradient[l][step_indicator < 0] = 0
                self.last_step_size[l] = self.step[l]

            return self.step

    def __init__(self, t_set, n, k, transformation=LTFArray.transform_id, combiner=LTFArray.combiner_xor, weights_mu=0, weights_sigma=1, weights_prng=RandomState(), logger=None):
        """
        Initialize a LTF Array Logistic Regression Learner for the specified LTF Array.

        :param t_set: The training set, i.e. a data structure containing challenge response pairs
        :param n: Input length
        :param k: Number of parallel LTFs in the LTF Array
        :param transformation: Input transformation used by the LTF Array
        :param combiner: Combiner Function used by the LTF Array (Note that not all combiner functions are supported by this class.)
        :param weights_mu: mean of the Gaussian that is used to choose the initial model
        :param weights_sigma: standard deviation of the Gaussian that is used to choose the initial model
        :param weights_prng: PRNG to draw the initial model from. Defaults to fresh `numpy.random.RandomState` instance.
        """
        self.iteration_count = 0
        self.training_set = t_set
        self.n = n
        self.k = k
        self.weights_mu = weights_mu
        self.weights_sigma = weights_sigma
        self.weights_prng = weights_prng
        self.iteration_limit = 10000
        self.convergence_decimals = 2
        self.sign_combined_model_responses = None
        self.sigmoid_derivative = full(self.training_set.N, None, double)
        self.min_distance = 1
        self.transformation = transformation
        self.combiner = combiner
        self.transformed_challenges = self.transformation(self.training_set.challenges, k)
        self.converged = False
        self.logger = logger

        assert self.n == len(self.training_set.challenges[0])

    @property
    def training_set(self):
        return self.__training_set

    @training_set.setter
    def training_set(self, val):
        self.__training_set = val

    def gradient(self, model):
        """
        Compute the gradient of the given model.

        :param model:
        :return:
        """

        # compute model responses
        model_responses = model.ltf_eval(self.transformed_challenges)
        combined_model_responses = self.combiner(model_responses)
        self.sign_combined_model_responses = sign(combined_model_responses)

        # cap the absolute value of this to avoid overflow errors
        MAX_RESPONSE_ABS_VALUE = 50
        combined_model_responses = sign(combined_model_responses) * \
                                   minimum(
                                       full(len(combined_model_responses), MAX_RESPONSE_ABS_VALUE, double),
                                       abs(combined_model_responses)
                                   )

        # compute the derivative from
        # the (-1,+1)-interval-sigmoid of combined model response on the all inputs
        # and the training set responses
        self.sigmoid_derivative = .5 * (2 / (1 + exp(-combined_model_responses)) - 1 - self.training_set.responses)
                                  # equivalent to self.set.responses * (1 - 1/(1 + exp(-self.set.responses * combined_model_responses)))

        def model_gradient_xor(l):
            #         Prod_i < w_i x_i >    /  < w_l x_l >          = Prod_(i \neq j)  < w_i x_i >
            return combined_model_responses / model_responses[:,l]

        def model_gradient_ip_mod2(l):
            if l % 2 == 0:  # for even l, the min operation takes place with the next value
                neighbor = model_responses[:,l+1]
            else:  # for odd l, the min operation takes place with the previous value
                neighbor = model_responses[:,l-1]

            max = amax((model_responses[:,l], neighbor), 0)

            return array([
                0
                if max[i] == neighbor[i] else
                combined_model_responses[i] / max[i]
                for i in range(self.training_set.N)
            ])
        # in a multiprocessing scenario the object references would not be the same!
        if compare_functions(self.combiner, LTFArray.combiner_xor):
            model_gradient = model_gradient_xor
        elif compare_functions(self.combiner, LTFArray.combiner_ip_mod2):
            model_gradient = model_gradient_ip_mod2
        else:
            raise Exception('No gradient function known for combiner %s' % self.combiner)

        ret = array([
            # sum over all challenges to the l-th Arbiter chain
            dot(
                self.sigmoid_derivative * model_gradient(l), # gradient
                self.transformed_challenges[:,l]  # all challenges to the l-th Arbiter chain
            )
            for l in range(self.k)
        ])
        return ret

    def learn(self):
        """
        Compute a model according to the given LTF Array parameters and training set.
        Note that this function can take long to return.
        :return: The computed model.
        """

        # log format
        def log_state():
            if self.logger is None:
                return
            self.logger.debug(
                '%i\t%f\t%f\t%s' % (
                    self.iteration_count,
                    distance,
                    norm(updater.step),
                    ','.join(map(str, model.weight_array.flatten()))
                )
            )

        # let numpy raise exceptions
        seterr(all='raise')

        # we start with a random model
        model = LTFArray(
            weight_array=LTFArray.normal_weights(self.n, self.k, self.weights_mu, self.weights_sigma, self.weights_prng),
            transform=self.transformation,
            combiner=self.combiner,
        )

        updater = self.RPropModelUpdate(model)
        converged = False
        distance = 1
        self.iteration_count = 0
        log_state()
        while not converged and distance > .01 and self.iteration_count < self.iteration_limit:
            self.iteration_count += 1

            # compute gradient & update model
            gradient = self.gradient(model)
            model.weight_array += updater.update(gradient)

            # check convergence
            converged = norm(updater.step) < 10**-self.convergence_decimals

            # check accuracy
            distance = (self.training_set.N - count_nonzero(self.training_set.responses == self.sign_combined_model_responses)) / self.training_set.N
            self.min_distance = min(distance, self.min_distance)

            # log
            log_state()

        if not converged and distance > .01:
            self.converged = False
        else:
            self.converged = True

        return model
