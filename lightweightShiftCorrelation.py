# -*- coding: utf-8 -*-
from pypuf import simulation, learner, tools
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.learner.regression.logistic_regression import LogisticRegression
import pypuf.tools

import numpy as np


#Apparently this is faster than numpy correlation
def vectorCorrelation(vectorA, vectorB):
    meanA = np.mean(vectorA)
    meanB = np.mean(vectorB)
    demeanedA = vectorA - meanA
    demeanedB = vectorB - meanB
    varA = np.dot(demeanedA, demeanedA)
    varB = np.dot(demeanedB, demeanedB)
    #varA = np.sum(demeanedA ** 2)
    #varB = np.sum(demeanedB ** 2)

    return np.dot(demeanedA, demeanedB) / (np.sqrt(varA)*np.sqrt(varB))

#--------------------------

numXors = 6
numChallenges = 500
numStages = 64

instance = LTFArray(
    weight_array=LTFArray.normal_weights(n=numStages+1, k=numXors),
    transform=LTFArray.transform_lightweight_secure_original,
    combiner=LTFArray.combiner_xor,

    bias =  True
)


challengeSet = tools.TrainingSet(instance=instance, N=numChallenges)
enableBias = True
transformOut = instance.transform(challengeSet.challenges, instance.k)
if enableBias:
    s = transformOut.shape 
    newTransformOut = np.ones((s[0], s[1] , s[2]+1)) 
    newTransformOut[:, :, :-1] = transformOut[:, :, :]
    transformOut = newTransformOut

#You could make the feature vectors mean free, then this step doesn't have to be performed in the correlation computation
#for currXor in range(numXors):
#    transformOut[:, currXor, :]   = (transformOut[:, currXor, :].T - np.mean(transformOut[:, currXor, :], axis=1)).T


 
shiftRawData = np.zeros((numXors, numStages, numChallenges, 2))
bestShiftData = np.zeros((numXors, 3))


shiftOverviewData = np.zeros((numXors, numXors, 2))


#I look at at each Arbiter separately:
#   -For each challenge, all shifted/rotated feature vectors of the current Arbiter are computed. Watch out: shiftedChallenge and currChallenges actually refer to feature vectors.
#   -Then each shifted feature vector of the current arbiter is looked at:
#       --We look for the arbiter  which the shifted feature vector has the highest correlation. I assume that a shifted feature vector only correlates with at most one other unshifted vector.
#   -

for currXor in range(numXors):

    shiftedChallenges = np.zeros((numChallenges, numStages, numStages+1))

    #Compute all possible shifted/rotated feature vectors for the currently selected Arbiter PUF
    for currentShift in range(numStages):
        shiftedChallenges[:, currentShift, :] = np.roll(transformOut[:, currXor, :], currentShift+1, axis = 1)

    

    #Determine which other Arbiter correlates the most with each shifted version of each feature vector for the current Arbiter.
    for currChallenge in range(numChallenges):
        if currChallenge % 100 == 0:
            print('-+-+--+-+-+-')
        bestShiftIndex = -1
        bestShiftXor = -1
        bestShiftCorr = 0
        for currentShift in range(numStages):
                   

            corrTmp = np.zeros((numXors,))
            for x in range(numXors):
                if x == currXor:
                    continue
                corrTmp[x] = vectorCorrelation(shiftedChallenges[currChallenge, currentShift, :], transformOut[currChallenge, x, :])

            maxCorr = np.max(np.abs(corrTmp))
            maxCorrXor = np.argmax(np.abs(corrTmp)) #The Arbiter index which has the highest correlation with the current shifted challenge

            shiftRawData[currXor, currentShift, currChallenge, 0] = maxCorrXor
            shiftRawData[currXor, currentShift, currChallenge, 1] = maxCorr
    print('*********************************')


    for currentShift in range(numStages):
        #Find the other Arbiter, that the most maximum correlation over all challenges for this shift
        bestXorForShift = np.argmax(np.bincount(shiftRawData[currXor, currentShift, :, 0].astype('int64')))
        #Determine the mean abs correlation for all feature vectors for which bestXorForShift has the highest correlation. 
        bestXorForShiftChallengeIndices = shiftRawData[currXor, currentShift, :, 0] == bestXorForShift
        meanAbsCorr = np.mean( np.abs(shiftRawData[currXor, currentShift, bestXorForShiftChallengeIndices, 1]) )

        #Keep track of the best shift for each combination of Arbiters
        if meanAbsCorr > shiftOverviewData[currXor, bestXorForShift, 1]: 
            shiftOverviewData[currXor, bestXorForShift, 0] = currentShift+1
            shiftOverviewData[currXor, bestXorForShift, 1] = meanAbsCorr



# Compute the acutal mean abs correlation over all feature vectors for the best shift for each combination of Arbiters
actualShiftCorrelation = np.zeros((numXors, numXors))
for x1 in range(numXors):

    for x2 in range(numXors):
        if x1 == x2: 
            continue
        corrs = np.zeros((numChallenges,))
        for currChallenge in range(numChallenges):
            shiftedChallenge = np.roll(transformOut[currChallenge, x1, :], shiftOverviewData[x1, x2, 0].astype('int64'), axis = 0)
            corrs[currChallenge] = vectorCorrelation(shiftedChallenge, transformOut[currChallenge, x2, :])
        actualShiftCorrelation[x1, x2] = np.mean(np.abs(corrs))



import scipy.io as scio


saveDict = {    'actualShiftCorrelation' : actualShiftCorrelation,
                'shiftOverviewData' : shiftOverviewData,
                'numXors' : numXors,
                'numChallenges' : numChallenges, 
                'numStages' : numStages
            }

#scio.savemat('lwShiftMatrix_' + str(numStages) + '_' + str(numXors), saveDict)
scio.savemat('lwShiftMatrix_' + str(numStages) + '_' + str(numXors), saveDict)





