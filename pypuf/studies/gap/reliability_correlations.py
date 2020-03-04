from numpy.random import RandomState
from numpy import copy, array
from pandas import DataFrame
from pypuf.simulation.arbiter_based.arbiter_puf import XORArbiterPUF
from pypuf.tools import sample_inputs

ns = [65, 129, 257, 513, 1025]
ks = [1, 2, 4, 8]
noisinesses = [0.05, 0.1, 0.2, 0.4, 0.8]
epsilons = [0.9, 1.0]
N = 10000
num = 100
reps = 100


def is_reliable(simulation, challenges, epsilon):
    responses = array([simulation.eval(challenges=challenges) for _ in range(reps)])
    return (abs(responses.sum(axis=0)) / reps) >= epsilon


df = DataFrame(columns=['n', 'k', 'noisiness', 'num', 'epsilon', 'equals'])

row = 0
for a, n in enumerate(ns):
    for b, k in enumerate(ks):
        for c, noisiness in enumerate(noisinesses):
            for d in range(num):
                puf = XORArbiterPUF(n=n, k=k, seed=d, transform='atf', noisiness=noisiness, noise_seed=d + 1000000)
                cs_plus = sample_inputs(n=n, num=N, random_instance=RandomState(d + 2000000))
                cs_plus[:, n//2] = 1
                cs_minus = copy(cs_plus)
                cs_minus[:, n//2] = -1
                for e, eps in enumerate(epsilons):
                    rs_plus = is_reliable(simulation=puf, challenges=cs_plus, epsilon=eps)
                    rs_minus = is_reliable(simulation=puf, challenges=cs_minus, epsilon=eps)
                    df.loc[row] = [n, k, noisiness, d, eps, sum(rs_plus == rs_minus) / N]
                    row += 1

            df = df.astype({'n': int, 'k': int, 'noisiness': float, 'num': int, 'epsilon': float, 'equals': float})
            df.to_csv(rf'results/gap.reliability_correlations_N={N}_reps={reps}.csv', header=True, index=False)
