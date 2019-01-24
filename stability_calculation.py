"""
This module provides the ability to calculate a list of stabilities for randomly chosen challenges of randomly chosen
MV XOR Arbiter PUF.
"""
from sys import argv, stderr
from numpy.random import RandomState
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, SimulationMajorityLTFArray, NoisyLTFArray
from pypuf import tools


def stability_figure_data(n, k, vote_count, sigma_noise_ratio, num, reps, random):
    """
    Returns a list of stabilities for randomly chosen challenges of randomly chosen MV XOR Arbiter PUF.
    :param n: Length of arbiter chains
    :param k: Number of arbiter chains
    :param vote_count: Number of votes for each chain
    :param sigma_noise_ratio: sigma_noise to sigma_model ratio of the arbiter chains
    :param num: number of challenges to compute stability for
    :param reps: number of samples per challenge to base the stability computation on
    :param random: random seed for all PRNG used here
    """
    sigma_noise = NoisyLTFArray.sigma_noise_from_random_weights(n, 1, sigma_noise_ratio)
    weights = LTFArray.normal_weights(n, k, random_instance=random)
    instance_mv = SimulationMajorityLTFArray(weights,
                                             LTFArray.transform_atf,
                                             LTFArray.combiner_xor,
                                             sigma_noise,
                                             random_instance_noise=random,
                                             vote_count=vote_count)

    stabilities = tools.approx_stabilities(instance_mv, num, reps, random)
    print('{' + ','.join(map(str, stabilities)) + '}')


if __name__ == "__main__":
    if len(argv) != 8:
        stderr.write('Stability Calculation for MV XOR Arbiter PUF\n')
        stderr.write('Usage:\n')
        stderr.write('stability_calculation.py n k r N reps seed\n')
        stderr.write('               n: number of bits per Arbiter chain\n')
        stderr.write('               k: number of Arbiter chains\n')
        stderr.write('               r: number of votes for majority vote\n')
        stderr.write('               sigma_noise_ratio: Ratio of sigma_model and sigma_noise\n')
        stderr.write('               N: number of challenges for sampling\n')
        stderr.write('               reps: number of repetitions per challenge for sampling\n')
        stderr.write('               seed: random seed\n')
        quit(1)

    stability_figure_data(int(argv[1]),
                          int(argv[2]),
                          int(argv[3]),
                          float(argv[4]),
                          int(argv[5]),
                          int(argv[6]),
                          RandomState(seed=int(argv[7], 16)))
