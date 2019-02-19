"""
Module for Learning Arbiter PUFs with Logistic Regression.
Heavily based on the work of Rührmair, Ulrich, et al. "Modeling attacks on physical unclonable functions." Proceedings
of the 17th ACM conference on Computer and communications security. ACM, 2010.
"""
from numpy import dtype, sign, dot, exp, array, seterr, minimum, full, amin, amax, array_split
from numpy import abs as np_abs
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
            """
            :param model: pypuf.simulation.arbiter_based.ltfarray.LTFArray
                          Model which is used to model a PUF.
            """
            self.model = model

        def update(self, gradient):
            """
            Use the gradient scaled with a constant to determine the update step.
            :param gradient: array of float
            :return: array of float
            """
            return -.3 * gradient

    class RPropModelUpdate(ModelUpdate):
        """
        Model update according to the Resilient Backpropagation algorithm. For details, see update() method.
        """

        def __init__(self, model, eta_minus=0.5, eta_plus=1.2):
            """
            :param model: pypuf.simulation.arbiter_based.ltfarray.LTFArray
            :param eta_minus: float
            :param eta_plus: float
            """
            self.n = n = model.n
            self.k = k = model.k

            self.eta_minus = eta_minus
            self.eta_plus = eta_plus
            self.delta_min = 10 ** -4
            self.delta_max = 10 ** +1
            self.last_gradient = full((k, n + 1), 1.0)
            self.last_step_size = full((k, n + 1), 0.0)
            self.step_size = full((k, n + 1), 1.0)
            self.step = full((k, n + 1), 0.0)
            self.step_size_max = full(self.n + 1, self.delta_max, dtype('float64'))
            self.step_size_min = full(self.n + 1, self.delta_min, dtype('float64'))

            super().__init__(model)

        def update(self, gradient):
            """
            Compute update step according to "Resilient Backpropagation" by
            Riedmiller, Martin, and Heinrich Braun. "A direct adaptive method for faster backpropagation learning:
            The RPROP algorithm."
            Neural Networks, 1993., IEEE International Conference on. IEEE, 1993.

            Implementation following the neat implementation used in
            Rührmair, Ulrich, et al. "Modeling attacks on physical unclonable functions."
            Proceedings of the 17th ACM conference on Computer and communications security.
            ACM, 2010.

            For their original code, please see http://www.pcp.in.tum.de/code/lr.zip,
            predictor.py:299
            :param gradient array of float
            :return: array of float
            """
            for l, grad_l in enumerate(gradient):
                step_indicator = sign(grad_l * self.last_gradient[l])

                self.step_size[l][step_indicator > 0] *= self.eta_plus
                self.step_size[l][step_indicator < 0] *= self.eta_minus

                self.step_size[l] = amin((self.step_size[l], self.step_size_max), 0)
                self.step_size[l] = amax((self.step_size[l], self.step_size_min), 0)

                self.step[l][step_indicator > 0] = -(
                    self.step_size[l][step_indicator > 0] * sign(grad_l[step_indicator > 0])
                )
                self.step[l][step_indicator < 0] = -self.last_step_size[l][step_indicator < 0]
                self.step[l][step_indicator == 0] = -self.step_size[l][step_indicator == 0] * sign(
                    grad_l[step_indicator == 0])

                self.last_gradient[l] = grad_l
                self.last_gradient[l][step_indicator < 0] = 0
                self.last_step_size[l] = self.step[l]

            return self.step

    def __init__(self, t_set, n, k, transformation=LTFArray.transform_id, combiner=LTFArray.combiner_xor, weights_mu=0,
                 weights_sigma=1, weights_prng=RandomState(), logger=None, iteration_limit=10000, minibatch_size=None,
                 convergance_decimals=2, shuffle=False):
        """
        Initialize a LTF Array Logistic Regression Learner for the specified LTF Array.

        :param t_set: The training set, i.e. a data structure containing challenge response pairs
        :param n: Input length
        :param k: Number of parallel LTFs in the LTF Array
        :param transformation: Input transformation used by the LTF Array
        :param combiner: Combiner Function used by the LTF Array
                         (Note that not all combiner functions are supported by this class.)
        :param weights_mu: mean of the Gaussian that is used to choose the initial model
        :param weights_sigma: standard deviation of the Gaussian that is used to choose the initial model
        :param weights_prng: PRNG to draw the initial model from. Defaults to fresh `numpy.random.RandomState` instance.
        :param logger: logging.Logger
                       Logger which is used to log detailed information of learn iterations.
        """
        self.iteration_count = 0
        self.epoch_count = 0
        self.gradient_step_count = 0
        self.training_set = t_set
        self.n = n
        self.k = k
        self.weights_mu = weights_mu
        self.weights_sigma = weights_sigma
        self.weights_prng = weights_prng
        self.iteration_limit = iteration_limit
        self.convergence_decimals = convergance_decimals
        self.transformation = transformation
        self.combiner = combiner
        self.efba_sub_challenges = None
        self.converged = False
        self.logger = logger
        self.updater = None
        self.minibatch_size = minibatch_size or self.training_set.N
        self.shuffle = shuffle

    @property
    def training_set(self):
        """
        This function returns the trainingset which is used to learn a PUF instance.
        :return: pypuf.tools.TrainingSet
        """
        return self.__training_set

    @training_set.setter
    def training_set(self, val):
        """
        Sets the traningset which is used to learn a PUF instance.
        :param val: pypuf.tools.TrainingSet
        """
        # pylint: disable-msg=W0201
        self.__training_set = val

    def gradient(self, model, efba_challenges_batch, responses_batch):
        """
        Compute the gradient of the given model.
        :param model: pypuf.simulation.arbiter_based.LTFArray
        :return: array of float
        """

        # compute model responses
        model_responses = model.core_eval(efba_challenges_batch)
        combined_model_responses = self.combiner(model_responses)

        # cap the absolute value of this to avoid overflow errors
        max_response_abs_value = 50
        max_response_abs_value_array = full(len(combined_model_responses), max_response_abs_value, dtype('float64'))
        combined_model_responses = sign(combined_model_responses) * minimum(max_response_abs_value_array,
                                                                            np_abs(combined_model_responses))

        # compute the derivative from
        # the (-1,+1)-interval-sigmoid of combined model response on the all inputs
        # and the training set responses
        sigmoid_derivative = .5 * (2 / (1 + exp(-combined_model_responses)) - 1 - responses_batch)

        # equivalent to self.set.responses * (1 - 1/(1 + exp(-self.set.responses * combined_model_responses)))

        def model_gradient_xor(l):
            """
            Caculates the gradient of the xored response at index l.
            :param l int
                     Index for weight array and challange array.
            :return array of float
            """
            #         Prod_i < w_i x_i >    /  < w_l x_l >          = Prod_(i \neq j)  < w_i x_i >
            return combined_model_responses / model_responses[:, l]

        def model_gradient_ip_mod2(l):
            """
            Caculates the gradient of the ip_mod2 combined responses at index l.
            :param l int
                     Index for weight array and challange array.
            :return array of float
            """
            if l % 2 == 0:  # for even l, the min operation takes place with the next value
                neighbor = model_responses[:, l + 1]
            else:  # for odd l, the min operation takes place with the previous value
                neighbor = model_responses[:, l - 1]

            maximum = amax((model_responses[:, l], neighbor), 0)

            return array([
                0
                if maximum[i] == neighbor[i] else
                combined_model_responses[i] / maximum[i]
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
                sigmoid_derivative * model_gradient(l),  # gradient
                efba_challenges_batch[:, l]  # all challenges to the l-th Arbiter chain
            )
            for l in range(self.k)
        ])
        return ret

    def learn(self, init_weight_array=None, eta_minus=0.5, eta_plus=1.2, refresh_updater=True):
        """
        Compute a model according to the given LTF Array parameters and training set.
        Note that this function can take long to return.
        :return: pypuf.simulation.arbiter_based.LTFArray
                 The computed model.
        """

        # log format
        def log_state():
            """
            This method is used to log a snapshot of learning variables while running.
            """
            if self.logger is None:
                return
            self.logger.debug(
                '%i\t%f\t%f\t%s' % (
                    self.iteration_count,
                    distance,
                    norm(self.updater.step),
                    ','.join(map(str, model.weight_array.flatten())),
                )
            )

        # let numpy raise exceptions
        seterr(all='raise')

        # Prepare challenges
        self.efba_sub_challenges = LTFArray.efba_bit(self.transformation(self.training_set.challenges, self.k))

        # we start with a random model
        model = LTFArray(
            weight_array=LTFArray.normal_weights(self.n, self.k, self.weights_mu, self.weights_sigma,
                                                 self.weights_prng),
            transform=self.transformation,
            combiner=self.combiner,
            bias=0.0,
        )

        if init_weight_array is not None:
            model.weight_array = init_weight_array

        if refresh_updater:
            self.updater = self.RPropModelUpdate(model, eta_minus=eta_minus, eta_plus=eta_plus)
        converged = False
        distance = 1
        self.iteration_count = 0
        log_state()
        number_of_batches = (self.training_set.N + 1) // self.minibatch_size
        efba_challenge_batches = []
        response_batches = []
        if not self.shuffle:
            efba_challenge_batches = array_split(self.efba_sub_challenges, number_of_batches)
            response_batches = array_split(self.training_set.responses, number_of_batches)
        while not converged and self.iteration_count < self.iteration_limit:
            self.iteration_count += 1
            self.epoch_count += 1

            if self.shuffle:
                if self.epoch_count > 1:
                    RandomState(seed=self.epoch_count).shuffle(self.efba_sub_challenges)
                    RandomState(seed=self.epoch_count).shuffle(self.training_set.responses)
                efba_challenge_batches = array_split(self.efba_sub_challenges, number_of_batches)
                response_batches = array_split(self.training_set.responses, number_of_batches)

            # compute gradient & update model
            for batch in range(number_of_batches):
                gradient = self.gradient(model, efba_challenge_batches[batch], response_batches[batch])
                model.weight_array += self.updater.update(gradient)
                self.gradient_step_count += 1

                # check convergence
                converged = norm(self.updater.step) < 10**-self.convergence_decimals

                # log
                log_state()

                if converged:
                    break

        if not converged:
            self.converged = False
        else:
            self.converged = True

        return model
