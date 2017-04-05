from numpy import sign, dot, around, exp, array, seterr, minimum, abs, full, count_nonzero, ones, zeros
from pypuf import simulation


class LogisticRegression():

    class ModelUpdate():

        def __init__(self, model):
            self.model = model

        def update(self, gradient):
            return -.3 * gradient

    class RPropModelUpdate(ModelUpdate):

        def __init__(self, model, eta_minus=0.5, eta_plus=1.2):

            n = model.n
            k = model.k

            self.eta_minus = eta_minus
            self.eta_plus = eta_plus
            self.last_gradient = full((k, n), 1)
            self.last_step_size = full((k, n), 0)
            self.step_size = full((k, n), 1)
            self.step = full((k, n), 0)

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

                self.step[l][step_indicator > 0] = -(self.step_size[l][step_indicator > 0] * sign(grad_l[step_indicator > 0]))
                self.step[l][step_indicator < 0] = -self.last_step_size[l][step_indicator < 0]
                self.step[l][step_indicator == 0] = -self.step_size[l][step_indicator == 0] * sign(grad_l[step_indicator == 0])

                self.last_gradient[l] = grad_l
                self.last_gradient[l][step_indicator < 0] = 0
                self.last_step_size[l] = self.step[l]

            return self.step

    def __init__(self, training_set, n, k, mu=0, sigma=1):
        self.set = training_set
        self.n = n
        self.k = k
        self.mu = 0
        self.sigma = 1
        self.iteration_limit = 100
        self.convergence_decimals = 3
        self.sign_combined_model_responses = None
        self.derivative = full(self.set.N, None)
        self.min_distance = 1

        assert self.n == len(training_set.challenges[0])

    def gradient(self, model):
        # compute model responses
        model_responses = model.ltf_eval(self.set.challenges)
        combined_model_responses = model.combiner_xor(model_responses)
        self.sign_combined_model_responses = sign(combined_model_responses)

        # cab the absolute value of this to avoid overflow errors
        MAX_RESPONSE_ABS_VALUE = 50
        combined_model_responses = sign(combined_model_responses) * \
                                   minimum(
                                       full(len(combined_model_responses), MAX_RESPONSE_ABS_VALUE),
                                       abs(combined_model_responses)
                                   )

        # compute the class probability(?) from
        # the (-1,+1)-interval-sigmoid of combined model response on the all inputs
        # and the training set responses
        self.derivative = .5 * (2 / (1 + exp(-combined_model_responses)) - 1 - self.set.responses)

        ret = array([
            dot(
                self.derivative
                *
                # estimated responses of the l-th LTF, derived from the real total value
                # and the (k-1) modelled values
                combined_model_responses / model_responses[:,l],
                self.set.challenges
            )
            for l in range(self.k)
        ])
        return ret

    def learn(self):
        # let numpy raise excpetions
        seterr(all='raise')

        # we start with a random model
        model = simulation.LTFArray(
            weight_array=simulation.LTFArray.normal_weights(self.n, self.k, self.mu, self.sigma),
            transform=simulation.LTFArray.transform_id,
            combiner=simulation.LTFArray.combiner_xor,
        )

        updater = self.RPropModelUpdate(model)

        i = 0
        converged = False
        distance = 1
        while not converged and distance > .01 and i < self.iteration_limit:
            i += 1

            # compute gradient & update model
            gradient = self.gradient(model)
            self.print_status(model)
            model.weight_array += updater.update(gradient)

            # check convergence
            converged = (
                [0] * self.n ==
                around(gradient, decimals=self.convergence_decimals)
            ).all()

            # check accuracy
            distance = (self.set.N - count_nonzero(self.set.responses == self.sign_combined_model_responses)) / self.set.N
            self.min_distance = min(distance, self.min_distance)

        if not converged and distance > .01:
            print('WARNING, MODEL DID NOT CONVERGE')

        return model

    def print_status(self, model):
        return # debug output disabled
        print('\nLR Training Set vs. Model Responses vs. Class Probability\n----------------------------------------------------------')
        for (idx,c) in enumerate(self.set.challenges):
            print(' '.join(['%+1d' % ci for ci in c]) +
                  ' => ' +
                  ('+1' if self.set.responses[idx] == 1 else '-1') +
                  '    ' +
                  '  '.join([('%+ 7.2f' % mr) for mr in model.ltf_eval([c])[0]]) +
                  '    ' +
                  ('% 10.5f' % self.class_probability[idx])
                  )

