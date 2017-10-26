
import scipy.io as sio
from pypuf import simulation, learner, tools
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.learner.regression.logistic_regression import LogisticRegression
import pypuf.tools
from itertools import permutations
import numpy as np


#Print the LR debug/ progress messages partially.
class Logger():

    def __init__(self, logInterval = 50):
        self.index = 0
        self.logInterval = logInterval

    def debug(self, debugstring):
        self.index = self.index  + 1
        if self.index >= self.logInterval:
            print(debugstring + '\n')

            self.index = 0






#Learning a lightweight PUF consists of 2 phases:
#   1.Find the initial model. This might take a few tries if we end up in local minima (acc ~ 0.50 ).
#     We end up with either with:
#       -a good model (acc > 0.95 or even higher) -> we can finish learning
#       -a model with mediocre accuracy (0.7 < acc < 0.95) -> we enter phase 2
#   2.We need to test possible permutations of the weight array to find the 'correct' permutation. To save
#     time we create a list of all accuracies on a validation set for all possible permutations. This list
#     indicates the chance of a permutation being the 'correct' one. We traverse the list from high to low 
#     accuracy and each time use the permuted weight array as the starting point for LR. We can expect to 
#     find the 'correct' permutation very early (we also should only <<100 RPROP iterations). This, however,
#     still needs to be quantified.

class LightweightMetaLearner():


    def __init__(self, puf_instance, training_set, validation_set, maxNumberInitialTrials = 5, maxNumberOptimizingTrials = 10, skipActualOptimizeLearning = False):

        self.puf_instance = puf_instance
        self.training_set = training_set
        self.validation_set = validation_set
        self.maxNumberOptimizingTrials = maxNumberOptimizingTrials
        self.maxNumberInitialTrials = maxNumberInitialTrials
        self.skipActualOptimizeLearning = skipActualOptimizeLearning


    #Returns the accuracy of model on the validation set
    def evalValidationSetAccuracy(self, model):

        return (self.puf_instance.eval(self.validation_set.challenges) == model.eval(self.validation_set.challenges)).sum() / self.validation_set.N

    #Returns whether the weight array of model is closely correlated to the original model. 
    def evalModelCorrelation(self, model):
        correlationList = []
        goodCorrelation = 1
        for x in range(self.puf_instance.k):
            correlationList.append(np.corrcoef(model.weight_array[x, :], self.puf_instance.weight_array[x,:])[0,1])
            if np.abs(correlationList[x]) < 0.95:
                goodCorrelation = 0
        return (goodCorrelation, correlationList)




    def learn(self):

        initialModelAccuracy = 0
        numOfInitialTrials = 0
        numOfOptiTrials = 0

        numXors = self.puf_instance.k

        #This .mat file was created in Matlab with a simple brute-force search. 
        loaded = sio.loadmat('shiftMatrix_5')
        #shiftMatrix is a 5x5 matrix. 
        #shiftMatrix[0, 1] = 32  means arbiter0 can take the place of arbiter1 if its feature vector is shifted by 32.
        shiftMatrix = loaded['shiftMatrix']



        # create the learner
        lr_learner = LogisticRegression(
            self.training_set,
            n=self.puf_instance.n,
            k=self.puf_instance.k,
            transformation=LTFArray.transform_lightweight_secure_original,
            combiner=LTFArray.combiner_xor,
            logger = Logger(),
            iteration_limit = 400
            )


        #--------initial trials
        #first we need a model with accuracy >> 50%
        print('---------------------------------------------------------')
        print('--------Start looking for an initial model---------------')
        print('---------------------------------------------------------')

        while initialModelAccuracy < 0.65 and numOfInitialTrials < self.maxNumberInitialTrials:

            initialModel = lr_learner.learn()
            initialModelAccuracy = self.evalValidationSetAccuracy(initialModel)
            numOfInitialTrials = numOfInitialTrials + 1


        goodCorrelation, correlationList = self.evalModelCorrelation(initialModel)

        startOptimizing = False
        if initialModelAccuracy > 0.95:
            print('The initial trials already leads to a good model fit (accuracy %f).' % (initialModelAccuracy))

            if goodCorrelation == 1:
                print('The initial model already fits the original weight matrix.')

        elif initialModelAccuracy < 0.58:
            print('We were only able to find a bad initial model (accuracy %f).' % (initialModelAccuracy))
        else:
            startOptimizing = True
            print('We found an initial model fit for optimizing (accuracy %f).' % (initialModelAccuracy))
        print(correlationList)


        #--------preparations for optimizing trials------------------------------------------------------------

        initalModelWeights = initialModel.weight_array

        
        perms = list(permutations(range(numXors)))
        perms = perms[1:] #remove identity permutation
        maxNumberOptimizingTrials = max(len(perms), self.maxNumberOptimizingTrials)

        #Create a ranking of all possible permutations based on their predictive performance on a validation set.
        #A higher accuracy implies a higher chance that it is the right permutation.
        accuracyList = []

        correctPermInitCorrList = []
        correctPermIndex = -1
        cnt = 0
        for currentPerm in perms:    
            weightMatrix = np.zeros(initalModelWeights.shape)
            for x in range(numXors):
                weightMatrix[currentPerm[x], :] = np.roll(initalModelWeights[x, :], shiftMatrix[x, currentPerm[x]])       
        
            initialModel.weight_array = weightMatrix
            modelAccuracy = self.evalValidationSetAccuracy(initialModel)
            accuracyList.append(modelAccuracy)
                   
            goodCorrelation, correlationList = self.evalModelCorrelation(initialModel)
            if np.sum(np.abs(correlationList) > 0.8) ==  self.puf_instance.k:
                correctPermIndex = cnt
                correctPermInitCorrList = correlationList
            cnt = cnt + 1



        sortedAccIndices = np.argsort(accuracyList )
        sortedAccIndices = sortedAccIndices[::-1] #get descending order
        correctPermAccuracyIndex = -1
        if correctPermIndex != -1:
            correctPermAccuracyIndex = np.where(sortedAccIndices == correctPermIndex)[0][0]
            print('The correct permutation should be found in trial #' + str(correctPermAccuracyIndex+1))
            print(correctPermInitCorrList)
        else:
            print('Surprisingly, we have not found the correct index for the permutation?!?')

        sortedAccuracyList = [accuracyList[index] for index in sortedAccIndices]

        print('We are going to learn shifted weight configurations that have the following accuracies: ' + str(sortedAccuracyList[:maxNumberOptimizingTrials]))


        optimizedModelAccuracy = initialModelAccuracy
        if startOptimizing and not self.skipActualOptimizeLearning:
            #--------optimizing trials----------------------------------------------------------------------
            #If the model has an accuracy < 0.99, we have to find the right permutation and optimize further
            print('---------------------------------------------------------')
            print('--------Start looking for an optimized model-------------')
            print('---------------------------------------------------------')


            lr_learner.iteration_limit = 50 #for optimizing we don't require a lot of iterations
        
            weightMatrix = np.zeros(initalModelWeights.shape)
            for currentIndex in sortedAccIndices:
                numOfOptiTrials = numOfOptiTrials + 1

                currentPerm = perms[currentIndex]

                for x in range(numXors):
                    weightMatrix[currentPerm[x], :] = np.roll(initalModelWeights[x, :], shiftMatrix[x, currentPerm[x]])
        
                print('---Starting next Training, Trial #' + str(numOfOptiTrials) + '---')
                initialModel.weight_array = weightMatrix
                goodCorrelation, correlationList = self.evalModelCorrelation(initialModel)                                          
                print('Pre-Optimizing Correlation: ' + str(correlationList))

                optimizedModel = lr_learner.learn(initWeightArray = weightMatrix)
                optimizedModelAccuracy = self.evalValidationSetAccuracy(optimizedModel)

                print('InitModel acc: %f\t Shifted Acc: %f\t New Acc: %f\n' % (initialModelAccuracy, accuracyList[currentIndex],  optimizedModelAccuracy))

                if optimizedModelAccuracy > initialModelAccuracy:
                    print('We have found a better accuracy (%f > %f).' % (optimizedModelAccuracy, initialModelAccuracy) )      
           
                    goodCorrelation, correlationList = self.evalModelCorrelation(optimizedModel)                                          
    
                    if goodCorrelation == 1:
                        print('We have found the original delay vector in Trial #' + str(numOfOptiTrials) + '.')            
                        break
                    else:
                        print('but it seems we have not found the original delay vector.')
                    print('Correlation List: ' + str(correlationList))   

        result = np.zeros((5,))
        result[0] = numOfInitialTrials
        result[1] = numOfOptiTrials
        result[2] = initialModelAccuracy
        result[3] = optimizedModelAccuracy
        result[4] = correctPermAccuracyIndex+1

        return  result



