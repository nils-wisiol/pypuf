"""
This module is a command line tool which provides the functionality to find the minimal number of votes which is
needed in order to satisfy the desired stability of challenges and overall desired stability of responses of the
SimulationMajorityLTFArray instance. The experiment is designed to find the number of votes for a range of k PUFs and
a defined number of restarts.

Example usage "python stab_c stab_all n k_max k_range s_ratio N restarts"

stab_c: float
        values: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        Desired stability of the challenges. The number which is used to decide whether a PUF is stable or not.
stab_all: float
          values: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
          Overall desired stability. The relative frequency of challenges which are greater equal than the
          desired_stability.
n: int
   values: [8, 16, 24, 32, 48, 64, 128]
   Number of bits per Arbiter chain
k_max: int
       Maximum number of Arbiter chains
k_range: int
         Number of step size between the number of Arbiter chains
s_ratio: float
         The noisiness factor which is used to scale sigma_noise. The value sigma_noise
         is the standard deviation of the noise distribution of the PUF instance simulation.
N: int
   The number of challenges which are used to evaluate the PUF.
restarts: int
          Number of restarts to the entire process
[log_name]: string default 'my_num_of_votes'
            Path to the main log file.
"""
from sys import stderr, argv
import argparse
from pypuf.experiments.experimenter import Experimenter
from pypuf.experiments.experiment.majority_vote import ExperimentMajorityVoteFindVotes
from pypuf.simulation.arbiter_based.ltfarray import LTFArray


def main(args):
    """
    This method starts several experiments in order to find the minimal number of votes required to satisfy an
    the desired stability of challenges.
    """
    parser = argparse.ArgumentParser(usage="Experiment to determine the minimum number of votes "
                                           "required to achieve a desired given stability.\n")
    parser.add_argument("stab_c", help="Desired stability of the challenges", type=float,
                        choices=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    parser.add_argument("stab_all", help="Overall desired stability", type=float,
                        choices=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    parser.add_argument("n", help="Number of bits per Arbiter chain", type=int,
                        choices=[8, 16, 24, 32, 48, 64, 128])
    parser.add_argument("k_max", help="Maximum number of Arbiter chains", type=int)
    parser.add_argument("k_range", help="Number of step size between the number of Arbiter chains", type=int,
                        choices=range(1, 33))
    parser.add_argument("s_ratio", help="Ratio of standard deviation of the noise and weights", type=float)
    parser.add_argument("N", help="Number of challenges to evaluate", type=int)
    parser.add_argument("restarts", help="Number of restarts to the entire process", type=int)
    parser.add_argument("--log_name", help="Path to the main log file.", type=str,
                        default='my_num_of_votes')
    args = parser.parse_args(args)

    if args.k_max <= 0:
        stderr.write("Negative maximum number of Arbiter chains")
        quit(1)

    seed_challenges = 0xf000
    iterations = 10
    n = args.n
    N = args.N

    # perform search for minimum number of votes required for each k
    experiments = []
    for i in range(args.restarts):
        for k in range(args.k_range, args.k_max + 1, args.k_range):
            log_name = args.log_name+'{0}'.format(k)
            exp = ExperimentMajorityVoteFindVotes(
                log_name=log_name,
                n=n,
                k=k,
                challenge_count=N,
                seed_instance=0xC0DEBA5E + i,
                seed_instance_noise=0xdeadbeef + i,
                transformation=LTFArray.transform_id,
                combiner=LTFArray.combiner_xor,
                mu=0,
                sigma=1,
                sigma_noise_ratio=args.s_ratio,
                seed_challenges=seed_challenges + i,
                desired_stability=args.stab_c,
                overall_desired_stability=args.stab_all,
                minimum_vote_count=1,
                iterations=iterations,
                bias=None
            )
            experiments.append(exp)

    experimenter = Experimenter(args.log_name, experiments)
    experimenter.run()

if __name__ == '__main__':
    main(argv[1:])
