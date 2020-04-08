from numpy import bool as np_bool, ones_like, triu, zeros, array
from pandas import read_csv, DataFrame, concat
from seaborn import set, heatmap
from matplotlib.pyplot import subplots, subplots_adjust, yticks


def similarity(dframe):
    length = len(dframe)
    cols = dframe.columns
    matrix = zeros(((len(cols),) * 2))
    for i, col1 in enumerate(cols):
        for j, col2 in enumerate(cols):
            matrix[i, j] = 1 - sum(abs(dframe[col1] - dframe[col2])) / length
    return DataFrame(data=matrix, columns=cols, index=cols, dtype=float)


def conditional_probs(df, n, ks, noisiness, eps):
    name = f'n={n}, ks={ks}, noisiness={noisiness}, eps={eps}'
    data = df[df['index'].str.startswith(name)]
    N = len(data)
    conditions = [
        (a, b, c, d, e, f)
        for a in range(2)
        for b in range(2)
        for c in range(2)
        for d in range(2)
        for e in range(2)
        for f in range(2)
    ]
    swaps = ['', '_inner', '_outer', '_left', '_right', '_big']
    targets = [prefix + swap
               for prefix in ['l_plus', 'l_minus', 'u']
               for swap in swaps]
    columns = ['condition'] + targets + ['frequency\nin %']
    result = DataFrame(columns=columns)
    for condition in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        string = '~'
        for value in condition:
            string += str(value)
        frequency = len(data[
                            (data['s'] == condition[0]) & (data['s_inner'] == condition[1])
                            ])
        line = [len(data[
                        (data[target] == 1) & (data['s'] == condition[0]) & (data['s_inner'] == condition[1])
                        ]) / frequency
                for target in targets
                ]
        line = [string] + line + [frequency / N]
        line_df = DataFrame(data=array([line]), columns=columns)
        result = concat(objs=[result, line_df])
    for condition in conditions:
        string = '~'
        for value in condition:
            string += str(value)
        frequency = len(data[
                         (data['s' + swaps[0]] == condition[0])
                         & (data['s' + swaps[1]] == condition[1])
                         & (data['s' + swaps[2]] == condition[2])
                         & (data['s' + swaps[3]] == condition[3])
                         & (data['s' + swaps[4]] == condition[4])
                         & (data['s' + swaps[5]] == condition[5])
                     ])
        line = [len(data[
                        (data[target] == 1)
                        & (data['s' + swaps[0]] == condition[0])
                        & (data['s' + swaps[1]] == condition[1])
                        & (data['s' + swaps[2]] == condition[2])
                        & (data['s' + swaps[3]] == condition[3])
                        & (data['s' + swaps[4]] == condition[4])
                        & (data['s' + swaps[5]] == condition[5])
                    ]) / frequency
                for target in targets
                ]
        line = [string] + line + [frequency / N]
        line_df = DataFrame(data=array([line]), columns=columns)
        result = concat(objs=[result, line_df])
    result.to_csv(path_or_buf=f'results/diverse_swaps_conditional_probs_noise={noisiness}_eps={eps}_ks={ks}.csv',
                  index=False,
                  )


def plot_similarities(df, n, k, noisiness, eps):
    set(style="white")
    fig, axes = subplots(nrows=1, ncols=1, figsize=(20, 18))
    name = f'n={n}, k={k}, noisiness={noisiness}, eps={eps}'
    data = df[df['index'].str.startswith(name)]
    corr = similarity(dframe=data.iloc[:, 1:])
    mask = triu(ones_like(corr, dtype=np_bool))
    for d in range(len(mask)):
        mask[d, d] = False
        num_filtered = sum(data[data.columns[d + 1]])
        corr.iloc[d, d] = num_filtered / len(data)
    subplots_adjust(top=0.9, bottom=0.15)
    heatmap(data=corr, ax=axes, cmap='RdYlGn', annot=True, vmin=0.0, vmax=1.0, mask=mask, square=True, linewidths=.5)
    axes.set_title(
        label=f'Matrix of coincidences of events regarding reliability (epsilon={eps}) using 5 different'
              '\n64-bit (8, 8)-iPUFs with 100k challenges per iPUF evaluated each 100 times.'
              '\nNote, that the diagonal shows the relative number of cases where the event is true.',
        fontsize=16,
    )
    fig.savefig(f'figures/cases_eps={eps}_similarity_heatmap.png', dpi=300)


def plot_conditional_probs():
    set(style="white")
    for eps in [0.9, 1.0]:
        for ks in [(1, 8), (8, 8)]:
            for noisiness in [0.05, 0.1]:
                df = read_csv(filepath_or_buffer=f'results/diverse_swaps_conditional_probs_noise={noisiness}'
                                                 f'_eps={eps}_ks={ks}.csv',
                              index_col=0,
                              )
                df['frequency\nin %'] *= 100

                fig, axes = subplots(nrows=1, ncols=1, figsize=(48, 12))
                subplots_adjust(bottom=0.1, top=0.9, left=0.15, right=1.03)
                heatmap(data=df.T, ax=axes, cmap='RdYlGn', annot=True, vmin=0.0, vmax=1.0, linewidths=.5)
                yticks(rotation=0)
                axes.set_ylabel('target event')
                axes.set_title(
                    label=f'Matrix of conditional probabilities of events regarding reliability'
                          f'\nusing 64-bit {ks}-iPUFs with 20k challenges per iPUF '
                          f'evaluated each 100 times (noisiness={noisiness}, epsilon={eps}).',
                    fontsize=14,
                )
                fig.savefig(f'figures/diverse_swaps_conditional_probs_noise={noisiness}_eps={eps}_ks={ks}.png', dpi=300)


for p1, p2, p3, p4 in [
    (n, ks, noisiness, epsilon)
    for n in [64]
    for ks in [(1, 8), (8, 8)]
    for noisiness in [0.05, 0.1]
    for epsilon in [0.9, 1.0]
]:
    p0 = read_csv(filepath_or_buffer=f'results/diverse_swaps_raw_rel_similarities.csv')
    # plot_similarities(df=p0, n=p1, k=p2, noisiness=p3, eps=p4)
    conditional_probs(df=p0, n=p1, ks=p2, noisiness=p3, eps=p4)
    pass

plot_conditional_probs()
