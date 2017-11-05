from numpy.random.mtrand import RandomState
from numpy import array, ndarray
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.tools import sample_inputs
import matplotlib.pyplot as plt


n = 24

instance = LTFArray(
    weight_array=LTFArray.normal_weights(n=n, k=1, random_instance=RandomState(seed=0xbeef)),
    transform=LTFArray.transform_atf,
    combiner=LTFArray.combiner_xor,
)


def plot_histogram(n):

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax = [ax]  # fix stupid matplotlib irregularity for x-by-1 plots
    fig.set_size_inches(w=16, h=9)
    legends = []
    for idx in range(1):
        ax[idx].set_title('delay values')
        ax[idx].set_xlabel('resulted accuracy')
        ax[idx].set_ylabel('frequency')
        ax[idx].set_xlim([-n, n])
        delays = instance.val(array(list(sample_inputs(n, 100000, random_instance=RandomState(seed=0xdead)))))
        ax[idx].hist(delays, bins=2*n, range=(-n, n), label='delay values')
        legends.append(ax[idx].legend(loc='center left', bbox_to_anchor=(1, 0.5)))
    if array([ axis.has_data() for axis in ax ]).any():
        fig.tight_layout()
        #fig.savefig(filename, bbox_extra_artists=legends, bbox_inches='tight')
        fig.show()
        input()


plot_histogram(n)