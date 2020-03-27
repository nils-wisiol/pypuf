from numpy import bool as np_bool, ones_like, triu, zeros
from pandas import read_csv, DataFrame
from seaborn import set, heatmap
from matplotlib.pyplot import subplots, subplots_adjust


def correlation(dataframe):
    length = len(dataframe)
    cols = dataframe.columns
    matrix = zeros(((len(cols),) * 2))
    for i, col1 in enumerate(cols):
        for j, col2 in enumerate(cols):
            matrix[i, j] = 1 - sum(abs(dataframe[col1] - dataframe[col2])) / length
    return DataFrame(data=matrix, columns=cols, index=cols, dtype=float)


set(style="white")

df = read_csv(filepath_or_buffer=f'results/raw_ipuf_reliability_correlations_swap.csv')

for n, k, noisiness, eps in [
    (n, k, noisiness, eps)
    for n in [64]
    for k in [8]
    for noisiness in [0.1]
    for eps in [0.9, 1.0]
]:
    fig, axes = subplots(nrows=1, ncols=1, figsize=(20, 18))
    name = f'n={n}, k={k}, noisiness={noisiness}, eps={eps}'
    data = df[df['index'].str.startswith(name)]
    corr = data.iloc[:, 1:].corr()  # correlation(dataframe=data.iloc[:, 1:])
    mask = triu(ones_like(corr, dtype=np_bool))
    for d in range(len(mask)):
        mask[d, d] = False
        num_filtered = sum(data[data.columns[d + 1]])
        corr.iloc[d, d] = num_filtered / len(data)
    subplots_adjust(top=0.9, bottom=0.15)
    heatmap(data=corr, ax=axes, cmap='RdYlGn', annot=True, vmin=-1.0, vmax=1.0, mask=mask, square=True, linewidths=.5)
    axes.set_title(
        label=f'Matrix of coincidences of events regarding reliability (epsilon={eps}) using 5 different'
              '\n64-bit (8, 8)-iPUFs with 100k challenges per iPUF evaluated each 100 times.'
              '\nNote, that the diagonal shows the relative number of cases where the event is true.',
        fontsize=16,
    )

    fig.savefig(f'figures/cases_eps={eps}_similarity_heatmap.png', dpi=300)
