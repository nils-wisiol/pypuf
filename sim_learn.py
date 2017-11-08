"""
This module is a command line tool which provides an interface for experiments which are designed to learn an arbiter
PUF LTFarray simulation with the logistic regression learning algorithm. If you want to use this tool you will have to
define nine parameters which define the experiment.
"""
import sys
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.experiments.experiment.logistic_regression import ExperimentLogisticRegression
from pypuf.experiments.experimenter import Experimenter


def main(args):
    """
    This method includes the main functionality of the module it parses the argument vector and executes the learning
    attempts on the PUF instances.
    """
    if len(args) < 10 or len(args) > 11:
        sys.stderr.write('LTF Array Simulator and Logistic Regression Learner\n')
        sys.stderr.write('Usage:\n')
        sys.stderr.write('sim_learn.py n k transformation combiner N restarts seed_instance seed_model [log_name]\n')
        sys.stderr.write('               n: number of bits per Arbiter chain\n')
        sys.stderr.write('               k: number of Arbiter chains\n')
        sys.stderr.write('  transformation: used to transform input before it is used in LTFs\n')
        sys.stderr.write('                  currently available:\n')
        sys.stderr.write('                  - id  -- does nothing at all\n')
        sys.stderr.write('                  - atf -- convert according to "natural" Arbiter chain\n')
        sys.stderr.write('                           implementation\n')
        sys.stderr.write('                  - mm  -- designed to achieve maximum PTF expansion length\n')
        sys.stderr.write('                           only implemented for k=2 and even n\n')
        sys.stderr.write('                  - lightweight_secure -- design by Majzoobi et al. 2008\n')
        sys.stderr.write('                                          only implemented for even n\n')
        sys.stderr.write('                  - shift_lightweight_secure -- design like Majzoobi\n')
        sys.stderr.write('                                                et al. 2008, but with the shift\n')
        sys.stderr.write('                                                operation executed first\n')
        sys.stderr.write('                                                only implemented for even n\n')
        sys.stderr.write('                  - soelter_lightweight_secure -- design like Majzoobi\n')
        sys.stderr.write('                                                  et al. 2008, but one bit different\n')
        sys.stderr.write('                                                  only implemented for even n\n')
        sys.stderr.write('                  - 1_n_bent -- one LTF gets "bent" input, the others id\n')
        sys.stderr.write('                  - 1_1_bent -- one bit gets "bent" input, the others id,\n')
        sys.stderr.write('                                this is proven to have maximum PTF\n')
        sys.stderr.write('                                length for the model\n')
        sys.stderr.write('                  - polynomial -- challenges are interpreted as polynomials\n')
        sys.stderr.write('                                  from GF(2^64). From the initial challenge c,\n')
        sys.stderr.write('                                  the i-th Arbiter chain gets the coefficients \n')
        sys.stderr.write('                                  of the polynomial c^(i+1) as challenge.\n')
        sys.stderr.write('                                  For now only challenges with length n=64 are accepted.\n')
        sys.stderr.write(
            '                  - permutation_atf -- for each Arbiter chain first a pseudorandom permutation \n')
        sys.stderr.write('                                       is applied and thereafter the ATF transform.\n')
        sys.stderr.write('                  - random -- Each Arbiter chain gets a random challenge derived from the\n')
        sys.stderr.write('                              original challenge using a PRNG.\n')
        sys.stderr.write('        combiner: used to combine the output bits to a single bit\n')
        sys.stderr.write('                  currently available:\n')
        sys.stderr.write('                  - xor     -- output the parity of all output bits\n')
        sys.stderr.write('                  - ip_mod2 -- output the inner product mod 2 of all output\n')
        sys.stderr.write('                               bits (even n only)\n')
        sys.stderr.write('               N: number of challenge response pairs in the training set\n')
        sys.stderr.write('        restarts: number of repeated initializations the learner\n')
        sys.stderr.write('       instances: number of repeated initializations the instance\n')
        sys.stderr.write('                  The number total learning attempts is restarts*instances.\n')
        sys.stderr.write('   seed_instance: random seed used for LTF array instance\n')
        sys.stderr.write('      seed_model: random seed used for the model in first learning attempt\n')
        sys.stderr.write('      [log_name]: path to the logfile which contains results from all instances. The tool '
                         'will add a ".log" to log_name. The default path is ./sim_learn.log\n')
        quit(1)

    n = int(args[1])
    k = int(args[2])
    transformation_name = args[3]
    combiner_name = args[4]
    N = int(args[5])
    restarts = int(args[6])

    instances = int(args[7])

    seed_instance = int(args[8], 16)
    seed_model = int(args[9], 16)

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

    log_name = 'sim_learn'
    if len(args) == 11:
        log_name = args[10]

    sys.stderr.write('Learning %s-bit %s XOR Arbiter PUF with %s CRPs and %s restarts.\n\n' % (n, k, N, restarts))
    sys.stderr.write('Using\n')
    sys.stderr.write('  transformation:       %s\n' % transformation)
    sys.stderr.write('  combiner:             %s\n' % combiner)
    sys.stderr.write('  instance random seed: 0x%x\n' % seed_instance)
    sys.stderr.write('  model random seed:    0x%x\n' % seed_model)
    sys.stderr.write('\n')

    # create different experiment instances
    experiments = []
    for j in range(instances):
        for start_number in range(restarts):
            l_name = '%s_%i_%i' % (log_name, j, start_number)
            experiment = ExperimentLogisticRegression(
                log_name=l_name,
                n=n,
                k=k,
                N=N,
                seed_instance=seed_instance + j,
                seed_model=seed_model + j + start_number,
                transformation=transformation,
                combiner=combiner
            )
            experiments.append(experiment)

    experimenter = Experimenter(log_name, experiments)
    # run the instances
    experimenter.run()

    # output format
    str_format = '{:<15}\t{:<10}\t{:<8}\t{:<8}\t{:<8}\t{:<8}\t{:<18}\t{:<15}\t{:<6}\t{:<8}\t{:<8}\t{:<8}'
    headline = str_format.format(
        'seed_instance', 'seed_model', 'i', 'n', 'k', 'N', 'trans', 'comb', 'iter', 'time', 'accuracy',
        'model_values\n'
    )
    # print the result headline
    sys.stderr.write(headline)

    log_file = open(log_name + '.log', 'r')

    # print the results
    result = log_file.readline()
    while result != '':
        sys.stderr.write(str_format.format(*result.split('\t')))
        result = log_file.readline()

    log_file.close()

if __name__ == '__main__':
    main(sys.argv)
