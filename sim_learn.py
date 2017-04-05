from numpy import random, amin, amax, mean, array, append
from pypuf import simulation, learner, tools
import time
import sys


n = 64
k = 2
N = 6000
restarts = 10
random_seed = 0xbeef

random.seed(random_seed) # reproduce 'random' numbers, remove to obtain different results

transformation = simulation.LTFArray.transform_id
combiner = simulation.LTFArray.combiner_xor

print('Learning %s-bit %s XOR Arbiter PUF with %s CRPs and %s restarts.\n' % (n, k, N, restarts))
print('Using')
print('  transformation: %s' % transformation)
print('  combiner:       %s' % combiner)
print('  random seed:    0x%x' % random_seed)
print('\n')


instance = simulation.LTFArray(
    weight_array=simulation.LTFArray.normal_weights(n, k),
    transform=transformation,
    combiner=combiner,
)

lr_learner = learner.LogisticRegression(
    tools.TrainingSet(instance=instance, N=N),
    n,
    k,
    transformation=transformation,
    combiner=combiner,
)

accuracy = array([])
training_times = array([])

for i in range(restarts):
    sys.stderr.write('\r%i/%i                 ' % (i+1, restarts))
    start = time.time()
    model = lr_learner.learn()
    end = time.time()
    training_times = append(training_times, end - start)
    dist = tools.approx_dist(instance, model, min(1000, 2 ** n))
    accuracy = append(accuracy, dist)
    #print('training time:                % 5.3fs' % (end - start))
    #print('min training distance:        % 5.3f' % lr_learner.min_distance)
    #print('test distance (1000 samples): % 5.3f\n' % dist)

print('\r              \r')
print('min/avg/max training time: % 5.3fs/% 5.3fs/% 5.3fs' % (amin(training_times), mean(training_times), amax(training_times)))
print('min/avg/max test accuracy: % 5.3f /% 5.3f /% 5.3f ' % (amin(1 - accuracy), mean(1 - accuracy), amax(1 - accuracy)))
print()
