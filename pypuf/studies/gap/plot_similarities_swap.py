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


def conditional_probs(df, n, k, noisiness, eps):
    name = f'n={n}, k={k}, noisiness={noisiness}, eps={eps}'
    data = df[df['index'].str.startswith(name)]
    targets = ['l_plus', 'l_minus', 'l_swap_plus', 'l_swap_minus', 'u', 'u_swap']
    columns = ['condition'] + targets
    conditions = ['s', 'not s', 's_swap', 'not s_swap', '00: s, s_swap', '01: s, s_swap', '10: s, s_swap',
                  '11: s, s_swap']
    result = DataFrame(columns=columns)
    for condition in conditions:
        line = ['00' if condition.startswith('00') else '01' if condition.startswith('01') else
                '10' if condition.startswith('10') else '11' if condition.startswith('11') else condition] + \
        [len(data[(data[target] == 1) & (data[condition.split('not ')[1]] == 0)]) /
         len(data[data[condition.split('not ')[1]] == 0]) if condition.startswith('not')
         else (len(data[(data[target] == 1) & (data[condition] == 1)]) / len(data[data[condition] == 1]))
         for target in targets]
        line_df = DataFrame(data=array([line]), columns=columns)
        result = concat(objs=[result, line_df])
    result.to_csv(path_or_buf=f'results/conditional_probs_eps={eps}.csv', index=False)


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
        df = read_csv(filepath_or_buffer=f'results/conditional_probs_eps={eps}.csv', index_col=0)

        fig, axes = subplots(nrows=1, ncols=1, figsize=(12, 8))
        subplots_adjust(bottom=0.1, top=0.9, left=0.15, right=1.03)
        heatmap(data=df.T, ax=axes, cmap='RdYlGn', annot=True, vmin=0.0, vmax=1.0, linewidths=.5)
        yticks(rotation=0)
        axes.set_ylabel('target event')
        axes.set_title(
            label=f'Matrix of conditional probabilities of events regarding reliability (epsilon={eps})'
                  '\nusing 5 different 64-bit (8, 8)-iPUFs with 100k challenges per iPUF evaluated each 100 times.'
                  '\nNote, that the diagonal shows the relative number of cases where the event is true.',
            fontsize=14,
        )
        fig.savefig(f'figures/conditional_probs_eps={eps}.png', dpi=300)


for p1, p2, p3, p4 in [
    (n, k, noisiness, eps)
    for n in [64]
    for k in [8]
    for noisiness in [0.1]
    for eps in [0.9, 1.0]
]:
    # p0 = read_csv(filepath_or_buffer=f'results/raw_ipuf_reliability_similarities_swap.csv')
    # plot_similarities(df=p0, n=p1, k=p2, noisiness=p3, eps=p4)
    # conditional_probs(df=p0, n=p1, k=p2, noisiness=p3, eps=p4)
    pass

plot_conditional_probs()
