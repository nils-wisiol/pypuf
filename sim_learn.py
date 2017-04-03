from numpy import random
from pypuf import simulation, learner, tools
import time


n = 64
k = 2
N = 6000
restarts = 3

print('Learning %s-bit %s XOR Arbiter PUF with %s CRPs and %s restarts.\n' % (n, k, N, restarts))

random.seed(0xbeef) # reproduce 'random' numbers, remove to obtain different results

instance = simulation.LTFArray(
    weight_array=simulation.LTFArray.normal_weights(n, k),
    transform=simulation.LTFArray.transform_id,
    combiner=simulation.LTFArray.combiner_xor,
)

lr_learner = learner.LogisticRegression(
    tools.TrainingSet(instance=instance, N=N),
    n,
    k,
)

for i in range(restarts):
    print('%i/%i ---------------------------------' % (i+1, restarts))
    start = time.time()
    model = lr_learner.learn()
    print('training time:                % 5.3fs' % (time.time() - start))
    print('min training distance:        % 5.3f' % lr_learner.min_distance)
    print('test distance (1000 samples): % 5.3f\n' % tools.approx_dist(instance, model, min(1000, 2**n)))
