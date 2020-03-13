from numpy.random import RandomState
from numpy import array, ndarray, concatenate, zeros, ones, sign, sum as np_sum, expand_dims, append
from pandas import DataFrame, concat

from pypuf.simulation.arbiter_based.arbiter_puf import InterposePUF
from pypuf.tools import sample_inputs, BIT_TYPE

ns = [64, 128]
ks = [1, 2, 4, 8]
noisinesses = [0.05, 0.1, 0.2, 0.4, 0.8]
epsilons = [0.9, 1.0]
N = 10000
num = 100
reps = 100


def is_reliable(simulation, challenges, epsilon):
    responses = array([simulation.eval(challenges=challenges) for _ in range(reps)])
    return (abs(responses.sum(axis=0)) / reps) >= epsilon


def interpose(challenges, bits, n2):
    if isinstance(bits, ndarray):
        N = challenges.shape[0]
        return concatenate((challenges[:, :n2], bits.reshape(N, 1), challenges[:, n2:]), axis=1)
    else:
        return concatenate(
            (
                challenges[:, :n2],
                zeros(shape=(challenges.shape[0], 1), dtype=BIT_TYPE) + bits,
                challenges[:, n2:]
            ), axis=1
        )


def majority_vote(simulation, challenges, repetitions):
    responses = zeros((repetitions, len(challenges)))
    for i in range(repetitions):
        responses[i, :] = simulation.eval(challenges)
    return sign(np_sum(a=responses, axis=0)) == 1


def truth_to_dec(truth_values):
    length = len(truth_values)
    return sum([int(truth_values[i]) * 2 ** (length - i - 1) for i in range(length)])


# l_plus reliable       l_minus reliable        u_plus      u reliable      s reliable
columns = ['num', '00000', '00001', '00010', '00011', '00100', '00101', '00110', '00111', '01000', '01001', '01010',
           '01011', '01100', '01101', '01110', '01111', '10000', '10001', '10010', '10011', '10100', '10101', '10110',
           '10111', '11000', '11001', '11010', '11011', '11100', '11101', '11110', '11111']

df = DataFrame(data=None, index=None, columns=columns, dtype=float)

for a, n in enumerate(ns):
    for b, k in enumerate(ks):
        for c, noisiness in enumerate(noisinesses):
            for d in range(num):
                ipuf = InterposePUF(
                    n=n,
                    k_down=k,
                    k_up=k,
                    seed=d,
                    transform='atf',
                    noisiness=noisiness,
                    noise_seed=d + 1000000,
                )
                cs = sample_inputs(n=n, num=N, random_instance=RandomState(d + 2000000))
                cs_plus = interpose(cs, +ones(shape=(N, 1), dtype=BIT_TYPE), n // 2)
                cs_minus = interpose(cs, -ones(shape=(N, 1), dtype=BIT_TYPE), n // 2)
                for e, eps in enumerate(epsilons):
                    cases = zeros(shape=32)
                    l_plus = is_reliable(simulation=ipuf.down, challenges=cs_plus, epsilon=eps)
                    l_minus = is_reliable(simulation=ipuf.down, challenges=cs_minus, epsilon=eps)
                    u = is_reliable(simulation=ipuf.up, challenges=cs, epsilon=eps)
                    u_plus = majority_vote(simulation=ipuf.up, challenges=cs, repetitions=99)
                    s = is_reliable(simulation=ipuf, challenges=cs, epsilon=eps)
                    for count in range(N):
                        cases[truth_to_dec([l_plus[count], l_minus[count], u[count], u_plus[count], s[count]])] += 1
                    line = DataFrame(
                        data=append([[d]], expand_dims(a=cases, axis=0) / N, axis=1),
                        index=[f'n={n}, k={k}, noisiness={noisiness}, eps={eps}'],
                        columns=columns,
                        dtype=float,
                    )
                    df = concat(objs=[df, line])
                    df.to_csv(path_or_buf=f'results/ipuf_reliability_correlations.csv')
