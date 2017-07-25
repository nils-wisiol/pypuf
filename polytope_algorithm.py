from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from numpy.random import RandomState
from pypuf.learner.base import Learner
from pypuf.learner.liu.partition import getChallenge
from pypuf.learner.liu.chebyshev import findCenter
from numpy import full, double, count_nonzero, array, ones, sum, sign, sqrt, minimum
from sys import stderr
from pypuf import tools
class PolytopeAlgorithm(Learner):
    def __init__(self,orig_LTFArray, t_set, n, k, transformation=LTFArray.transform_id, combiner=LTFArray.combiner_xor, weights_mu=0, weights_sigma=1, weights_prng=RandomState()):
        """
        Initialize a LTF Array Polytope Learner for the specified LTF Array.

        :param t_set: The training set, i.e. a data structure containing challenge response pairs
        :param n: Input length
        :param k: Number of parallel LTFs in the LTF Array
        :param transformation: Input transformation used by the LTF Array
        :param combiner: Combiner Function used by the LTF Array (Note that not all combiner functions are supported by this class.)
        :param weights_mu: mean of the Gaussian that is used to choose the initial model
        :param weights_sigma: standard deviation of the Gaussian that is used to choose the initial model
        :param weights_prng: PRNG to draw the initial model from. Defaults to fresh `numpy.random.RandomState` instance.
        """
        self.orig_LTFArray=orig_LTFArray
        self.iteration_count = 0
        self.__training_set = t_set
        self.n = n
        self.k = k
        self.weights_mu = weights_mu
        self.weights_sigma = weights_sigma
        self.weights_prng = weights_prng
        self.iteration_limit = 150
        self.convergence_decimals = 3
        self.sign_combined_model_responses = None
        self.sigmoid_derivative = full(self.training_set.N, None, double)
        self.min_distance = 1
        self.transformation = transformation
        self.combiner = combiner
        self.transformed_challenges = self.transformation(self.training_set.challenges, k)

        assert self.n == len(self.training_set.challenges[0])

    @property
    def training_set(self):
        return self.__training_set

    @training_set.setter
    def training_set(self, val):
        self.__training_set = val

    
    def learn(self):
        model = LTFArray(
            weight_array=LTFArray.normal_weights(self.n, self.k, self.weights_mu, self.weights_sigma, self.weights_prng),
            transform=self.transformation,
            combiner=self.combiner,
        )
        self.iteration_count = 0
        challenges=[]
        responses=[]
        
#        challenges.append(array([1,1,-1,1,1]))
#        responses.append(1)
#        challenges.append(array([1,1,1,1,1]))
#        responses.append(1)
#        challenges.append(array([-1,1,-1,1,1]))
#        responses.append(-1)
#        challenges.append(array([1,-1,-1,1,1]))
#        responses.append(-1)
#        challenges.append(array([-1,-1,1,1,-1]))
#        responses.append(-1)
        challenges.append(ones(self.n))
        responses.append(self.__signum(sum(self.orig_LTFArray.weight_array*challenges[0])))
        print(challenges)
        print(responses)
        print("")
        print("originales array")
        print(self.orig_LTFArray.weight_array)
        print("-------------------------------------------")
        memo_radius=0;
        while self.iteration_count < self.iteration_limit:
            
            self.updateModel(model)
#            print(challenges)
#            print(responses)
#            print("hello")
            stderr.write('\riter %5i         ' % (self.iteration_count))
            self.iteration_count += 1
            (center,radius) = self.__chebyshev_center(challenges, responses)
            model.weight_array=[center]
            # check accuracy
            #distance = (self.training_set.N - count_nonzero(self.training_set.responses == self.sign_combined_model_responses)) / self.training_set.N
            distance = tools.approx_dist(self.orig_LTFArray, model, min(10000, 2 ** model.n))
            self.min_distance = min(distance, self.min_distance)
            
#            print("distance %f"%distance)
#            print("radius %f"%radius)
#            print("center:")
#            print(center)
            if (distance < 0.01):
                break
            if (radius==memo_radius):
                print("radius not changing")
            memo_radius=radius
            minAccuracy=abs(radius*sqrt(model.n))

            newC=self.__closest_challenge(center, minAccuracy);
            challenges.append(newC)
#            print("")
#            print("challenge:")
#            print(newC)
            responses.append(self.__signum(sum(newC*self.orig_LTFArray.weight_array)))
#            print(self.orig_LTFArray.weight_array)
#            print(model.weight_array)
        print("")
        print("----------------------------")        
        print(model.weight_array)
        print("----------------------------")
        return model
    
    def __chebyshev_center(self,challenges,responses):
        
        return findCenter(challenges,responses)
            
    
    def __closest_challenge(self, center, minAccuracy):
        return getChallenge(center, minAccuracy)
    
         
    def __signum(self,x):
        if sign(x)!=0:
            return sign(x)
        else:
            return -1
        
    def updateModel(self,model):
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