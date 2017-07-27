from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from numpy.random import RandomState
from pypuf.learner.base import Learner
from pypuf.learner.liu.partition import getChallenge
from pypuf.learner.liu.chebyshev import findCenter
from pypuf.learner.liu.chebyshev2 import findCenter2
from pypuf.learner.liu.simplex import AdjustedSimplexAlg
from numpy import full, double, ones, sum, sign, sqrt, minimum
from pypuf import tools
from sys import stderr


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
        self.iteration_limit = 10000
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
        

        challenges.append(ones(self.n))
        responses.append(self.__signum(sum(self.orig_LTFArray.weight_array*challenges)))
        
        while self.iteration_count < self.iteration_limit:
            
            self.__updateModel(model)
            stderr.write('\riter %5i         \n' % (self.iteration_count))
            self.iteration_count += 1
            [center,radius] = self.__chebyshev_center(challenges, responses)
            stderr.write("radius ")
            stderr.write("%f\n"%radius)
            stderr.write("distance ")
            
            model.weight_array=[center]
            distance = tools.approx_dist(self.orig_LTFArray, model, min(10000, 2 ** model.n))
            self.min_distance = min(distance, self.min_distance)
            if (distance < 0.01):
                break
            minAccuracy=abs(radius*sqrt(model.n))
            stderr.write("%f\n"%distance)
            newC=self.__closest_challenge(center, minAccuracy);
            challenges.append(newC)
            responses.append(self.__signum(sum(newC*self.orig_LTFArray.weight_array)))

        return model
    
    def __chebyshev_center(self,challenges,responses):
        #simplex=AdjustedSimplexAlg()
        #[cOwn,rOwn]= simplex.solve(challenges,responses)
        [cNormal,rNormal] =findCenter(challenges,responses)
        #stderr.write("own=\n")
        #stderr.write("%f"%rOwn)
        #simplex.printSol(cOwn)
        #stderr.write("normal=\n")
        #stderr.write("%f"%rNormal)
        #simplex.printSol(cNormal)
        
        return [cNormal,rNormal]
        #return [cOwn,rOwn]
            
    
    def __closest_challenge(self, center, minAccuracy):
        return getChallenge(center, minAccuracy)
        #challenge=zeros(self.n)
        #for i in range(self.n):
        #    challenge[i]=self.__signum(random.random()-0.5)
        #return challenge;
        
    def __signum(self,x):
        if sign(x)!=0:
            return sign(x)
        else:
            return -1
        
    def __updateModel(self,model):
        model_responses = model.ltf_eval(self.transformed_challenges)
        combined_model_responses = self.combiner(model_responses)
        self.sign_combined_model_responses = sign(combined_model_responses)
        MAX_RESPONSE_ABS_VALUE = 50
        combined_model_responses = sign(combined_model_responses) * \
                                   minimum(
                                       full(len(combined_model_responses), MAX_RESPONSE_ABS_VALUE, double),
                                       abs(combined_model_responses)
                                   )