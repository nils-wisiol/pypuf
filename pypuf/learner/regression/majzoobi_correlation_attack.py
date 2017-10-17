from numpy import array, mean, concatenate, split
from numpy.linalg import norm
from numpy.random import RandomState
from pypuf.learner.regression.logistic_regression import LogisticRegression
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from copy import deepcopy

hardcoded = [0.202589,0.105608,-0.089875,-0.020000,-0.027314,-0.127932,-0.064410,0.142692,-0.144838,0.294124,0.110442,-0.024509,-0.031952,0.230188,-0.143699,-0.135333,-0.051677,-0.175955,0.127949,0.069484,0.160321,-0.236036,-0.108658,-0.016871,-0.035111,0.005779,0.089029,-0.375303,0.147801,0.226992,0.452826,0.328093,0.066109,-0.383705,0.021449,-0.055674,-0.375258,-0.074147,0.003847,0.006415,-0.132562,-0.141065,-0.258261,-0.076499,0.271578,0.137844,-0.350741,0.038583,0.006573,-0.081283,0.257779,-0.096707,-0.172485,0.075669,-0.212963,0.150255,0.213628,0.233590,0.126094,-0.130296,-0.005459,0.140637,-0.158246,0.085998,-0.039725,-0.121904,0.146068,-0.005576,0.294468,-0.252938,0.237967,0.048633,0.154533,-0.167989,0.532480,-0.034741,0.126497,0.086676,-0.294796,-0.040014,-0.182778,0.045969,-0.258278,0.122672,0.037463,0.098419,-0.075045,0.037870,0.115656,-0.176630,-0.015313,0.061615,-0.017910,0.138778,0.247792,-0.196544,-0.309173,-0.024724,-0.093924,0.151941,0.244177,-0.246638,-0.113798,0.316776,-0.049111,0.156390,0.006885,-0.052075,0.064814,-0.150732,-0.253363,-0.015386,-0.028859,0.121110,0.001359,-0.147807,0.037251,-0.246305,0.383088,0.007456,0.002246,0.465251,0.059411,0.083436,0.095333,0.137737,0.067327,0.106777
]


class NoMappingsFoundException(Exception):
    def __init__(self):
        super().__init__()


class Rotation:
    def __init__(self, source, target, rotate, dist, invert):
        self.source = source
        self.target = target
        self.rotate = rotate
        self.dist = dist
        self.invert = invert

    def __str__(self):
        return "Shift {0} -> {1}, Rotate {2}, Dist={3}".format(
            self.source,
            ("-" if self.invert else "") + str(self.target), self.rotate, self.dist)


class MajzoobiCorrelationAttack(LogisticRegression):
    def __init__(self, t_set, n, k, transformation=LTFArray.transform_id, combiner=LTFArray.combiner_xor,
                 weights_mu=0, weights_sigma=1, weights_prng=RandomState(), logger=None):
        super().__init__(t_set, n, k, transformation, combiner, weights_mu, weights_sigma, weights_prng, logger)
        self.good_model_vector = None
        self.best_model_dist = 1

    def learn(self, weight_init=None):
        model = super().learn()
        rotated_model = deepcopy(model)

        iterations = 0
        model_weights_norm = self.norm_flat_weights(model)

        if self.distance <= 0.1 and self.distance < self.best_model_dist:
            self.good_model_vector = model_weights_norm
            self.best_model_dist = self.distance
        elif 0.05 < self.distance < 0.4 and self.good_model_vector is not None:
            try:
                rotated = self._rotate_weight_vector(self.good_model_vector, model_weights_norm)
                rotated_weights = array(split(rotated, self.k))
                iterations += self.iteration_count
                rotated_model = super().learn(rotated_weights)
                if self.distance <= 0.1 and self.distance < self.best_model_dist:
                    self.good_model_vector = self.norm_flat_weights(rotated_model)
                    self.best_model_dist = self.distance
            except NoMappingsFoundException:
                pass
        return model, rotated_model

    def norm_flat_weights(self, model):
        model_weights_norm = array([])
        for l in range(self.k):
            model_weights_norm = concatenate(
                (model_weights_norm, model.weight_array[l] / norm(model.weight_array[l]))
            )
        return model_weights_norm

    def _rotate_weight_vector(self, original, shifted):
        rotations = self._align(original, shifted)

        shifted_fixed = self._apply_rotations(rotations, shifted)
        interval = self.n // 2

        for l in range(len(shifted) // interval):
            start = l * interval
            original_array = original[start:start + interval]
            shifted_fixed[start:start + interval] = self._fix_rotation(
                original_array, shifted_fixed[start:start + interval])

        return array(shifted_fixed)

    @staticmethod
    def _rotate(l, i):
        return concatenate((l[i:], l[:i]))

    def _align(self, original, shifted):
        assert len(original) == len(shifted)
        k = len(original) // self.n
        window_width = self.n // 2
        rotations = [[None for _ in range(k)] for _ in range(k)]
        for puf_no_original in range(k):
            original_slice = original[puf_no_original * self.n:puf_no_original * self.n + self.n]
            for puf_no_shifted in range(k):
                shifted_slice = shifted[puf_no_shifted * self.n:puf_no_shifted * self.n + self.n]
                for rotation in range(self.n):
                    rotated = self._rotate(shifted_slice, rotation)
                    for window_start in range(self.n):
                        window_end = (window_start + window_width) % self.n
                        if window_start < window_end:
                            original_window = original_slice[window_start:window_end]
                            rotated_window = rotated[window_start:window_end]
                        else:
                            original_window = concatenate((original_slice[window_start:], original_slice[:window_end]))
                            rotated_window = concatenate((rotated[window_start:], rotated[:window_end]))
                        dist = norm(original_window - rotated_window)
                        dist_invert = norm(original_window + rotated_window)
                        dist_min = min(dist, dist_invert)

                        invert = dist_invert < dist
                        rot = Rotation(puf_no_shifted, puf_no_original, rotation, dist_min, invert)
                        if rotations[puf_no_shifted][puf_no_original] is None:
                            rotations[puf_no_shifted][puf_no_original] = rot
                        else:
                            if rot.dist < rotations[puf_no_shifted][puf_no_original].dist:
                                rotations[puf_no_shifted][puf_no_original] = rot

        ranked_rotations = []

        for source in range(k):
            for target in range(k):
                if rotations[source][target] is not None:
                    ranked_rotations.append(rotations[source][target])

        ranked_rotations.sort(key=lambda x: x.dist)
        sources = [0] * k
        target = [0] * k

        rotations_filtered = []

        for rot in ranked_rotations:
            #print(rot)
            if sources[rot.source] == 1 or target[rot.target] == 1:
                continue
            rotations_filtered.append(rot)
            sources[rot.source] = 1
            target[rot.target] = 1

        if sum(sources) < k or sum(target) < k:
            raise NoMappingsFoundException()

        return rotations_filtered

    def _apply_rotations(self, rotations, weights):
        rotated_weights = [None] * len(weights)
        for item in rotations:
            assert item is not None
            source = self._rotate(weights[item.source * self.n:item.source * self.n + self.n], item.rotate)
            if item.invert:
                source = [-x for x in source]
            rotated_weights[item.target * self.n:item.target * self.n + self.n] = source
        return rotated_weights

    @classmethod
    def _fix_rotation(cls, original, shifted):
        assert len(original) == len(shifted)
        minimum_dist = float('inf')
        best_rotation = 0
        for i in range(len(original)):
            rotated = cls._rotate(shifted, i)
            dist = norm(original - rotated)
            if dist < minimum_dist:
                minimum_dist = dist
                best_rotation = i
        return cls._rotate(shifted, best_rotation)
