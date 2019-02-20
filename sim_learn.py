"""
This module is a command line tool which provides an interface for experiments which are designed to learn an arbiter
PUF LTFarray simulation with the logistic regression learning algorithm. If you want to use this tool you will have to
define nine parameters which define the experiment.
"""
import sys
import argparse
from pypuf.experiments.experimenter import Experimenter
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.experiments.experiment.logistic_regression import ExperimentLogisticRegression


def main(args):
    """
    This method includes the main functionality of the module it parses the argument vector and executes the learning
    attempts on the PUF instances.
    """
    parser = argparse.ArgumentParser(
        prog='sim_learn',
        description="LTF Array Simulator and Logistic Regression Learner",
    )
    parser.add_argument("n", help="number of bits per Arbiter chain", type=int)
    parser.add_argument("k", help="number of Arbiter chains", type=int)
    parser.add_argument(
        "transformation",
        help="used to transform input before it is used in LTFs. Currently available: "
             '"atf,id",'
             '"lightweight_secure",'
             '"permutation_atf",'
             '"polynomial,random",'
             '"shift",'
             '"soelter_lightweight_secure"',
        type=str,
    )
    parser.add_argument(
        'combiner',
        help='used to combine the output bits to a single bit. Currently available: "ip_mod2", "xor"',
        type=str,
    )
    parser.add_argument('N', help='number of challenge response pairs in the training set', type=int)
    parser.add_argument('restarts', help='number of repeated initializations the learner', type=int)
    parser.add_argument(
        'instances',
        help='number of repeated initializations the instance\n'
             'The number total learning attempts is restarts*instances.',
        type=int,
    )
    parser.add_argument('seed_instance', help='random seed used for LTF array instance', type=str)
    parser.add_argument('seed_model', help='random seed used for the model in first learning attempt', type=str)
    parser.add_argument(
        '--log_name',
        help='path to the logfile which contains results from all instances. The tool '
             'will add a ".log" to log_name. The default path is ./sim_learn.log',
        default='sim_learn',
        type=str,
    )
    parser.add_argument(
        '--seed_challenges',
        help='random seed used to draw challenges for the training set',
        type=str,
    )
    parser.add_argument('--seed_distance', help='random seed used to calculate the accuracy', type=str)

    args = parser.parse_args(args)

    n = args.n
    k = args.k
    transformation_name = args.transformation
    combiner_name = args.combiner
    N = args.N
    restarts = args.restarts

    instances = args.instances

    seed_instance = int(args.seed_instance, 16)
    seed_model = int(args.seed_model, 16)

    seed_challenges = 0x5A551
    if args.seed_challenges is not None:
        seed_challenges = int(args.seed_challenges, 16)
    seed_distance = 0xB055
    if args.seed_distance is not None:
        seed_distance = int(args.seed_distance, 16)

    transformation = None
    combiner = None

    try:
        transformation = getattr(LTFArray, 'transform_%s' % transformation_name)
    except AttributeError:
        sys.stderr.write('Transformation %s unknown or currently not implemented\n' % transformation_name)
        quit()

    try:
        combiner = getattr(LTFArray, 'combiner_%s' % combiner_name)
    except AttributeError:
        sys.stderr.write('Combiner %s unknown or currently not implemented\n' % combiner_name)
        quit()

    log_name = args.log_name

    sys.stderr.write('Learning %s-bit %s XOR Arbiter PUF with %s CRPs and %s restarts.\n\n' % (n, k, N, restarts))
    sys.stderr.write('Using\n')
    sys.stderr.write('  transformation:       %s\n' % transformation)
    sys.stderr.write('  combiner:             %s\n' % combiner)
    sys.stderr.write('  instance random seed: 0x%x\n' % seed_instance)
    sys.stderr.write('  model random seed:    0x%x\n' % seed_model)
    sys.stderr.write('\n')

    # create different experiment instances
    experimenter = Experimenter(log_name)
    for j in range(instances):
        for start_number in range(restarts):
            l_name = '%s_%i_%i' % (log_name, j, start_number)
            experiment = ExperimentLogisticRegression(
                progress_log_prefix=l_name,
                n=n,
                k=k,
                N=N,
                seed_instance=seed_instance + j,
                seed_model=seed_model + j + start_number,
                transformation=transformation,
                combiner=combiner,
                seed_challenge=seed_challenges,
                seed_chl_distance=seed_distance,
            )
            experimenter.queue(experiment)

    # run the instances
    experimenter.run()


if __name__ == '__main__':
    main(sys.argv[1:])
