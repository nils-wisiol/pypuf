from numpy.random import RandomState
from numpy import copy
from pandas import DataFrame
from pypuf.simulation.arbiter_based.arbiter_puf import XORArbiterPUF
from pypuf.tools import sample_inputs

ns = [65, 129, 257, 513, 1025]
ks = [1, 2, 3, 4, 5, 6, 7, 8]
noisinesses = [0.05, 0.1, 0.2, 0.4, 0.8]
epsilons = [0.9, 1.0]
N = 100000
num = 1000


def is_reliable(simulation, challenges, epsilon):
    responses = simulation.eval(challenges=challenges)
    return responses >= epsilon


df = DataFrame(columns=['n', 'k', 'noisiness', 'num', 'epsilon', 'equals'])
row = 0

for a, n in enumerate(ns):
    for b, k in enumerate(ks):
        for c, noisiness in enumerate(noisinesses):
            for d in range(num):
                for e, eps in enumerate(epsilons):
                    puf = XORArbiterPUF(n=n, k=k, seed=d, transform='atf', noisiness=noisiness, noise_seed=d + 1000000)
                    cs_plus = sample_inputs(n=n, num=N, random_instance=RandomState(d + 2000000))
                    cs_plus[:, n//2] = 1
                    cs_minus = copy(cs_plus)
                    cs_minus[:, n//2] = -1
                    rs_plus = is_reliable(simulation=puf, challenges=cs_plus, epsilon=eps)
                    rs_minus = is_reliable(simulation=puf, challenges=cs_minus, epsilon=eps)
                    df.loc[row] = [n, k, noisiness, d, eps, sum(rs_plus == rs_minus)]
                    row += 1

df = df.astype({'n': int, 'k': int, 'noisiness': float, 'num': int, 'epsilon': float, 'equals': int})
df.to_csv(r'results/gap.reliability_correlations.csv', header=True)
