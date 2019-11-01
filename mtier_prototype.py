from scipy.stats import pearsonr
from numpy import array, copy, ones, zeros, shape, empty

from pypuf.tools import TrainingSet, ChallengeResponseSet, approx_dist
from pypuf.learner.regression.logistic_regression import LogisticRegression
from pypuf.simulation.arbiter_based.arbiter_puf import InterposePUF
from pypuf.simulation.arbiter_based.ltfarray import LTFArray


def extend_tset(tset, pos):
    N, n = shape(tset)
    xtset = empty(shape=(N*2, n+1))
    xtset[:N, :pos] = tset[:, :pos]
    xtset[:N, :pos] = tset[:, :pos]
    xtset[N:, pos+1:] = tset[:, pos:]
    xtset[N:, pos+1:] = tset[:, pos:]
    xtset[N:, pos] = 1
    xtset[:N, pos] = -1
    return xtset


# Set Parameters
n = 64
n_2 = n//2
k_up = 2
k_down = 6
# seed = 42
N = int(800e3)
transform = LTFArray.transform_atf

print('\nLearn {}-bit ({}, {})-IPUF with {} CRPs using a multi-tier learning process based on logistic regression...\n'
      .format(n, k_up, k_down, N))

# Initialize IPUF, training set and learner
print('\nInitialize IPUF, training set and the learner for down PUF...')
ipuf = InterposePUF(n=n, k_up=k_up, k_down=k_down, transform=transform)
tset = TrainingSet(instance=ipuf, N=N)
xtset = extend_tset(tset=tset, pos=n_2)
learner_down = LogisticRegression(t_set=xtset, n=n+1, k=k_down, transformation=transform)

# Learn down PUF
print('\nLearn down PUF...')
model_down = learner_down.learn()
print('total epochs: {}\t\tconverged: {}'.format(learner_down.epoch_count, learner_down.converged))

first_half_correlations = array([[pearsonr(model_down.weight_array[j, :n_2], ipuf.down.weight_array[i, :n_2])[0]
                                  for i in range(k_down)] for j in range(k_down)])
last_half_correlations = array([[pearsonr(model_down.weight_array[j, n_2:], ipuf.down.weight_array[i, n_2+1:])[0]
                                 for i in range(k_down)] for j in range(k_down)])
print('correlations:')
print('first half:')
print(abs(first_half_correlations).max(axis=1))
print('last half:')
print(abs(last_half_correlations).max(axis=1))
print('##################################################################')
print(abs(model_down.weight_array[:, :n_2]).mean())
print(abs(model_down.weight_array[:, -n_2:]).mean())
print('##################################################################')

# Include middle weight into model_down
print('\nInclude middle weight into model_down...')
weights_extended = zeros(shape=(k_down, n + 2))
weights_extended[:, :n_2] = model_down.weight_array[:, :n_2]
weights_extended[:, n_2+1:] = model_down.weight_array[:, n_2:]
model_down.weight_array = weights_extended
model_down.n += 1
accuracy_down = 1 - approx_dist(instance1=ipuf.down, instance2=model_down, num=10**4)
print('accuracy down: {}'.format(accuracy_down))

"""
# Filter CRPs in order to learn down PUF
print('\nFilter CRPs in order to learn down PUF...')
indices = []
responses = []
cs_extended_p = ones(shape=(N, n+1))
cs_extended_m = copy(a=cs_extended_p)
cs_extended_m[:, n//2] = -1
for i in range(N):
    rp, rm = model_down.eval(array([cs_extended_p[i], cs_extended_m[i]]))
    if rp == rm:
        continue
    r = tset.responses[i]
    indices.append(i)
    if rp == r:
        responses.append(1)
    else:
        responses.append(-1)
print('number of filtered CRPs: {}'.format(len(indices)))

print('\nInitialize the training set and the learner for up PUF...')
tset_filtered = ChallengeResponseSet(challenges=tset.challenges[indices], responses=array(responses))
learner_up = LogisticRegression(t_set=tset_filtered, n=n, k=k_up, transformation=transform)

# Learn up PUF
print('\nLearn up PUF...')
model_up = learner_up.learn()
print('\ntotal epochs: {}\t\tconverged: {}'.format(learner_up.epoch_count, learner_up.converged))
first_half_correlations = array([[pearsonr(model_up.weight_array[j, :n//2], ipuf.up.weight_array[i, :n//2])[0]
                                  for i in range(k_up)] for j in range(k_up)])
last_half_correlations = array([[pearsonr(model_up.weight_array[j, -(n//2):], ipuf.up.weight_array[i, -(n//2):])[0]
                                 for i in range(k_up)] for j in range(k_up)])
print('\ncorrelations:')
print('\nfirst half:')
print(abs(first_half_correlations).max())
print('\nlast half:')
print(abs(last_half_correlations).max())
accuracy_up = 1 - approx_dist(instance1=ipuf.up, instance2=model_up, num=10**4)
print('accuracy up: {}'.format(accuracy_up))


print('\nFINISH')
"""
