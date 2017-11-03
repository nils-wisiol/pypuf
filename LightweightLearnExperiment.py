# -*- coding: utf-8 -*-
from pypuf import simulation, learner, tools
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.learner.regression.logistic_regression import LogisticRegression
import pypuf.tools
from  LightweightMetaLearner import LightweightMetaLearner

import numpy as np

import datetime




numXors = 5
numStages = 64
numAttackedInstances = 200
trainingSetSize = 100000

filename = ('experimentResults_stages_' + str(numStages) + '_xors_' + str(numXors)
            + '_trainSetSize_' + str(trainingSetSize) +  '_time_{:%Y_%m_%d__%H_%M_%S}'.format(datetime.datetime.now()) )
print(filename)



results = np.zeros((numAttackedInstances, 5))
with open(filename, 'wb') as f:
    np.save(f, results)

for iteration in range(numAttackedInstances):
    print('+++++++++++++++ Running Experiment Iteration #' + str(iteration) + ' ++++++++++++++++-')
    instance = LTFArray(
        weight_array=LTFArray.normal_weights(n=65, k=4),
        transform=LTFArray.transform_lightweight_secure_original,
        combiner=LTFArray.combiner_xor,
        bias =  True
    )

    #If you want to actually simulate the attack set skipActualOptimizeLearning to False!
    meta = LightweightMetaLearner(instance, training_set = tools.TrainingSet(instance=instance, N=trainingSetSize), 
                                  validation_set = tools.TrainingSet(instance=instance, N=5000), maxNumberOptimizingTrials = -1,
                                  skipActualOptimizeLearning = True)
    #numOfInitialTrials, numOfOptiTrials, initialModelAccuracy, optimizedModelAccuracy  
    results[iteration, :] =  meta.learn()
    print('--- ' + str(results[iteration, 4]))
    with open(filename, 'wb') as f:
        np.save(f, results)

print(results)