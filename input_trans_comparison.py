from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from sys import stderr
from pypuf.experiments.experiment.logistic_regression import ExperimentLogisticRegression
from pypuf.experiments.experimenter import Experimenter
from random import sample
from string import ascii_uppercase
import argparse


def run_input_trans_comparison(
        n,
        k,
        transformations,
        Ns,
        combiner=LTFArray.combiner_xor,
        instance_sample_size=100,
        seed_instance=0x15eed,
        initial_seed_model=0x5eed,
        log_file_prefix = None,
        iteration_limit=1000,
        ):
    """
    This function runs experiments to compare different input transformations regarding their
    influence on the learnability by Logistic Regression.
    For a given size of LTFArray (n,k), and a given list of input transformations, and a given
    list of number of CRPs in the training set (Ns), for each combination of input transformation
    and training set size, an LTF Array instance is created. Each LTF Array will be learned using
    instance_sample_size attempts with different initializations.
    All LTF Arrays are using the given combiner function. All results are written the log files
    with the given prefix.
    """

    # Set up logging
    if log_file_prefix is None:
        log_file_prefix = ''.join(sample(list(ascii_uppercase), 5))
    stderr.write('log file prefix: %s\n' % log_file_prefix)
    stderr.write('running %s experiments' % str(len(transformations)*len(Ns)*instance_sample_size))

    # Experimenter instance that is used for experiment scheduling
    e = Experimenter(log_file_prefix, [])

    # Add all transformation/N-combinations to the Experimenter Queue
    for transformation in transformations:
        for N in Ns:
            seed_model = initial_seed_model
            for _ in range(instance_sample_size):
                e.experiments.append(
                    ExperimentLogisticRegression(
                        log_name=log_file_prefix,
                        n=n,
                        k=k,
                        N=N,
                        seed_model=seed_model,
                        seed_instance=seed_instance,
                        transformation=transformation,
                        combiner=combiner,
                        iteration_limit=iteration_limit,
                    )
                )
                seed_model += 1
                seed_instance += 1

    # Run all experiments
    e.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-n", type=int, help="Number of stages/bits.")
    parser.add_argument("-k", type=int, help="Number of Arbiter chains.")
    parser.add_argument("-s", "--samples", type=int, help="Instance sample size.")
    parser.add_argument("-m", "--modelseed", type=str, help="Initial seed of the model as hex value (e.g. \"0xbeef\").")
    parser.add_argument("-t", "--transformations", nargs="+", type=str,
                        required=True, help="<Required> List of transformations")
    parser.add_argument("-N", "--crpcount", nargs="+", type=int,
                        required=True, help="<Required> List of CRP counts.")
    parser.add_argument("-i", "--iteration-limit", type=int, default=1000,
                        help="Cancel learning after this number of iterations.")
    parser.add_argument("-h", "--help", action='help', default=argparse.SUPPRESS,
                        help="Usage: python3 input_trans_comparison.py 64 2 10 \"0xbeef\" "
                             "-t \"id\" \"polynomial\" -N 10 100")
    args = parser.parse_args()

    given_transformations = []
    for t in args.transformations:
        try:
            given_transformations.append(getattr(LTFArray, 'transform_%s' % t))
        except AttributeError:
            stderr.write('Transformation %s unknown or currently not implemented\n' % t)
            quit()

    run_input_trans_comparison(
        n=args.n,
        k=args.k,
        transformations=given_transformations,
        Ns=args.crpcount,
        instance_sample_size=args.samples,
        initial_seed_model=int(args.modelseed, 16),
        iteration_limit=int(args.iteration_limit),
    )
