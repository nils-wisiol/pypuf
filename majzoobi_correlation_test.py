from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.learner.regression.majzoobi_correlation_attack import MajzoobiCorrelationAttack
from numpy.random import RandomState
from numpy import array, concatenate, round
from numpy.linalg import norm
from pypuf import tools
from time import time

n = 64
k = 4
N = 12000
target_dist = 0.01
instance_seed = RandomState(0xdeadbeef)

transformation = LTFArray.transform_lightweight_secure_original
combiner = LTFArray.combiner_xor

instance = LTFArray(
    weight_array=LTFArray.normal_weights(n, k, random_instance=instance_seed),
    transform=transformation,
    combiner=combiner,
)
weights = instance.weight_array
weights_norm = array([])
for l in range(k):
    weights_norm = concatenate((
        (weights_norm, weights[l] / norm(weights[l])))
    )
#print(','.join(['%f' % x for x in weights_norm]))

weight_seed = RandomState(0xdead)
regression = MajzoobiCorrelationAttack(tools.TrainingSet(instance=instance, N=N), n, k, transformation, combiner,
                                       weights_prng=weight_seed)

# print(regression.rotate_weight_vector(array(original), array(shifted)))

start_time = time()
lr_finished = False
lr_finish_time = 0
mz_finished = False
mz_finish_time = 0


while not (lr_finished and mz_finished):
    model, rotated_model = regression.learn()
    # print(','.join(['%f' % x for x in model.weight_array.flatten()]))
    # print(','.join(['%f' % x for x in rotated_model.weight_array.flatten()]))
    lr_dist = tools.approx_dist(instance, model, 50000)
    mz_dist = tools.approx_dist(instance, rotated_model, 50000)
    if lr_dist <= target_dist and not lr_finished:
        lr_finished = True
        lr_finish_time = time()
        print("LR finished!")
    if mz_dist <= target_dist and not mz_finished:
        mz_finished = True
        mz_finish_time = time()
        print("MZ finished!")

    print(lr_dist, mz_dist)

print()
print("n=%d, k=%d" % (n, k))
print("LR finished in %d seconds" % round(lr_finish_time - start_time))
print("MZ finished in %d seconds" % round(mz_finish_time - start_time))
