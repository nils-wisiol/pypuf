from numpy.random import RandomState
from numpy import array, ndarray, concatenate, zeros, ones, sign, sum as np_sum, copy
from pandas import DataFrame, concat

from pypuf.simulation.arbiter_based.arbiter_puf import InterposePUF
from pypuf.tools import sample_inputs, BIT_TYPE


def is_reliable(simulation, challenges, epsilon):
    responses = array([simulation.eval(challenges=challenges) for _ in range(reps)])
    return (abs(responses.sum(axis=0)) / reps) >= epsilon


def interpose(challenges, bits, position):
    if isinstance(bits, ndarray):
        N = challenges.shape[0]
        return concatenate((challenges[:, :position], bits.reshape(N, 1), challenges[:, position:]), axis=1)
    else:
        return concatenate(
            (
                challenges[:, :position],
                zeros(shape=(challenges.shape[0], 1), dtype=BIT_TYPE) + bits,
                challenges[:, position:]
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


ns = [64]
ks = [8]
noisinesses = [0.1]
epsilons = [0.9, 1.0]
N = 100000
num = 100
reps = 100


columns = ['l_plus', 'l_minus', 'u', 'u_plus', 's', 'l_swap_plus', 'l_swap_minus', 'u_swap', 'u_swap_plus', 's_swap']
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
                pos = n // 2
                cs_swap = copy(cs)
                cs_swap[:, pos - 1:pos] = -cs_swap[:, pos - 1:pos]
                cs_plus = interpose(challenges=cs, bits=+ones(shape=(N, 1), dtype=BIT_TYPE), position=pos)
                cs_minus = interpose(challenges=cs, bits=-ones(shape=(N, 1), dtype=BIT_TYPE), position=pos)
                cs_swap_plus = interpose(challenges=cs_swap, bits=+ones(shape=(N, 1), dtype=BIT_TYPE), position=pos)
                cs_swap_minus = interpose(challenges=cs_swap, bits=-ones(shape=(N, 1), dtype=BIT_TYPE), position=pos)

                for e, eps in enumerate(epsilons):
                    s = is_reliable(
                        simulation=ipuf,
                        challenges=cs,
                        epsilon=eps,
                    )
                    s_swap = is_reliable(
                        simulation=ipuf,
                        challenges=cs_swap,
                        epsilon=eps,
                    )
                    l_plus = is_reliable(
                        simulation=ipuf.down,
                        challenges=cs_plus,
                        epsilon=eps,
                    )
                    l_minus = is_reliable(
                        simulation=ipuf.down,
                        challenges=cs_minus,
                        epsilon=eps,
                    )
                    u = is_reliable(
                        simulation=ipuf.up,
                        challenges=cs,
                        epsilon=eps,
                    )
                    u_plus = majority_vote(
                        simulation=ipuf.up,
                        challenges=cs,
                        repetitions=reps,
                    )
                    l_swap_plus = is_reliable(
                        simulation=ipuf.down,
                        challenges=cs_swap_plus,
                        epsilon=eps,
                    )
                    l_swap_minus = is_reliable(
                        simulation=ipuf.down,
                        challenges=cs_swap_minus,
                        epsilon=eps,
                    )
                    u_swap = is_reliable(
                        simulation=ipuf.up,
                        challenges=cs_swap,
                        epsilon=eps,
                    )
                    u_swap_plus = majority_vote(
                        simulation=ipuf.up,
                        challenges=cs_swap,
                        repetitions=reps,
                    )
                    for j in range(N):
                        unequal = s[j] == s_swap[j]
                        events = array(
                            [[l_plus[j], l_minus[j], u[j], u_plus[j], s[j], l_swap_plus[j], l_swap_minus[j], u_swap[j],
                              u_swap_plus[j], s_swap[j]]],
                            dtype=bool,
                        )
                        line = DataFrame(
                            data=events,
                            index=[f'n={n}, k={k}, noisiness={noisiness}, eps={eps}, ipuf={d}, num={j}'],
                            columns=columns,
                            dtype=bool,
                        )
                        df = concat(objs=[df, line])
                    df.to_csv(path_or_buf=f'results/raw_ipuf_reliability_similarities_swap.csv')
