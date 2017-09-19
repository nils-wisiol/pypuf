# -*- coding: utf-8 -*-
from pypuf import simulation, learner, tools
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.learner.regression.logistic_regression import LogisticRegression
import pypuf.tools



class Logger():
    def debug(self, debugstring):
        print(debugstring + '\n')


# create a simulation with random (Gaussian) weights
# for 64-bit 4-XOR 
#instance = LTFArray(
#    weight_array=LTFArray.normal_weights(n=64, k=4),
#    transform=LTFArray.transform_lightweight_secure,
#    combiner=LTFArray.combiner_xor,
#)


instance = LTFArray(
    weight_array=LTFArray.normal_weights(n=65, k=4),
    transform=LTFArray.transform_lightweight_secure_original,
    combiner=LTFArray.combiner_xor,
    bias =  True
)

from  LightweightMetaLearner import *
meta = LightweightMetaLearner(instance, training_set = tools.TrainingSet(instance=instance, N=40000), validation_set = tools.TrainingSet(instance=instance, N=5000))

meta.learn()

# learn and test the model
#model = lr_learner.learn()
#accuracy = 1 - tools.approx_dist(instance, model, 1000)

# output the result
#print('Learned a 64bit 2-xor XOR Arbiter PUF from 12000 CRPs with accuracy %f' % accuracy)