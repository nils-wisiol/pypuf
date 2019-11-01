from scipy.stats import pearsonr
from numpy import array, copy, ones, zeros

from pypuf.tools import TrainingSet, ChallengeResponseSet, approx_dist
from pypuf.learner.regression.logistic_regression import LogisticRegression
from pypuf.simulation.arbiter_based.arbiter_puf import InterposePUF
from pypuf.simulation.arbiter_based.ltfarray import LTFArray


# Set Parameters
n = 64
n_2 = n//2
k_up = 8
k_down = 1
transform = 'atf'
combiner = 'xor'

"""
mean = 0
num = 100
for i in range(num):
    ipuf = InterposePUF(n=n, k_up=k_up, k_down=k_down, transform=transform)

    down_weights = zeros(shape=(k_down, n + 1))
    down_weights[:, :n_2] = ipuf.down.weight_array[:, :n_2]
    down_weights[:, n_2+1:] = ipuf.down.weight_array[:, n_2+1:-1]
    down_puf = LTFArray(
        weight_array=down_weights,
        transform=transform,
        combiner=combiner,
        bias=ipuf.down.weight_array[:, -1]
    )

    accuracy_down = 1 - approx_dist(instance1=down_puf, instance2=ipuf.down, num=10**4)
    mean += accuracy_down

mean /= num
print(mean)
"""


mean = 0
num = 100
for i in range(num):
    ipuf = InterposePUF(n=n, k_up=k_up, k_down=k_down, transform=transform)

    down_weights = zeros(shape=(k_down, n))
    down_weights[:, :n_2] = ipuf.down.weight_array[:, :n_2]
    down_weights[:, n_2:] = ipuf.down.weight_array[:, n_2+1:-1]
    down_puf = LTFArray(
        weight_array=down_weights,
        transform=transform,
        combiner=combiner,
        bias=ipuf.down.weight_array[:, -1]
    )

    accuracy_down = 1 - approx_dist(instance1=down_puf, instance2=ipuf, num=10**4)
    print(accuracy_down)
    mean += accuracy_down

# mean /= num
# print(mean)
