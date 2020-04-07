from numpy.random import RandomState
from numpy import array, ndarray, concatenate, zeros, ones, sign, abs as abs_np, sum as sum_np, copy, delete
from pandas import DataFrame, concat

from pypuf.simulation.arbiter_based.arbiter_puf import InterposePUF
from pypuf.tools import sample_inputs, BIT_TYPE


def is_reliable(simulation, challenges, repetitions=11, epsilon=0.9):
    responses = array([simulation.eval(challenges=challenges) for _ in range(repetitions)])
    axis = 0
    return abs_np(sum_np(responses, axis=axis)) / (2 * responses.shape[axis]) + 0.5 >= epsilon


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
    return sign(sum_np(a=responses, axis=0)) == 1


def truth_to_dec(truth_values):
    length = len(truth_values)
    return sum([int(truth_values[i]) * 2 ** (length - i - 1) for i in range(length)])


ns = [64]
ks = [(1, 8), (8, 8)]
noisinesses = [0.05, 0.1]
epsilons = [0.9, 1.0]
N = 20000
num = 3
reps = 11


columns = ['l_plus', 'l_minus', 'u', 'u_plus', 's',
           'l_plus_inner', 'l_minus_inner', 'u_inner', 'u_plus_inner', 's_inner',
           'l_plus_outer', 'l_minus_outer', 'u_outer', 'u_plus_outer', 's_outer',
           'l_plus_left', 'l_minus_left', 'u_left', 'u_plus_left', 's_left',
           'l_plus_right', 'l_minus_right', 'u_right', 'u_plus_right', 's_right',
           'l_plus_big', 'l_minus_big', 'u_big', 'u_plus_big', 's_big',
           ]
df = DataFrame(data=None, index=None, columns=columns, dtype=float)

for a, n in enumerate(ns):
    for b, (k_up, k_down) in enumerate(ks):
        for c, noisiness in enumerate(noisinesses):
            for d in range(num):
                ipuf = InterposePUF(
                    n=n,
                    k_down=k_down,
                    k_up=k_up,
                    seed=d,
                    transform='atf',
                    noisiness=noisiness,
                    noise_seed=d + 1000000,
                )
                cs = sample_inputs(n=n, num=N, random_instance=RandomState(d + 2000000))
                pos = n // 2
                for e, eps in enumerate(epsilons):
                    events = zeros((1, N), dtype=bool)
                    for positions in [(), (pos - 1, pos), (pos - 2, pos + 1), (pos - 2, pos), (pos - 1, pos + 1),
                                      (pos - 2, pos - 1, pos, pos + 1)]:
                        cs_swap = copy(cs)
                        for p in positions:
                            cs_swap[:, p] = -cs_swap[:, p]
                        cs_plus = interpose(challenges=cs, bits=+ones(shape=(N, 1), dtype=BIT_TYPE), position=pos)
                        cs_minus = interpose(challenges=cs, bits=-ones(shape=(N, 1), dtype=BIT_TYPE), position=pos)
                        cs_plus_swap = interpose(challenges=cs_swap, bits=+ones(shape=(N, 1), dtype=BIT_TYPE), position=pos)
                        cs_minus_swap = interpose(challenges=cs_swap, bits=-ones(shape=(N, 1), dtype=BIT_TYPE),
                                                  position=pos)
                        s = is_reliable(
                            simulation=ipuf,
                            challenges=cs_swap,
                            epsilon=eps,
                        )
                        l_plus = is_reliable(
                            simulation=ipuf.down,
                            challenges=cs_plus_swap,
                            epsilon=eps,
                        )
                        l_minus = is_reliable(
                            simulation=ipuf.down,
                            challenges=cs_minus_swap,
                            epsilon=eps,
                        )
                        u = is_reliable(
                            simulation=ipuf.up,
                            challenges=cs_swap,
                            epsilon=eps,
                        )
                        u_plus = majority_vote(
                            simulation=ipuf.up,
                            challenges=cs_swap,
                            repetitions=reps,
                        )
                        events = concatenate(
                            seq=(events, array([l_plus, l_minus, u, u_plus, s], dtype=bool)),
                            axis=0,
                        )
                    events = delete(arr=events, obj=0, axis=0)
                    block = DataFrame(
                        data=events.T,
                        index=[
                            f'n={n}, ks={(k_up,k_down)}, noisiness={noisiness}, eps={eps}, ipuf={d}, num={j}'
                            for j in range(N)
                        ],
                        columns=columns,
                        dtype=bool,
                    )
                    df = concat(objs=[df, block])
                    df.to_csv(path_or_buf=f'results/raw_rel_similarities_swap_diverse.csv')
