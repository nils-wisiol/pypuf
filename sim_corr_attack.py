"""
This module is a command line tool which provides an interface for experiments which are designed to learn an arbiter
PUF LTFarray simulation with the logistic regression learning algorithm. If you want to use this tool you will have to
define nine parameters which define the experiment.
"""
from sys import argv, stderr
from pypuf.experiments.experimenter import Experimenter
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf.experiments.experiment.correlation_attack import ExperimentCorrelationAttack


def main(args):
    """
    This method includes the main functionality of the module it parses the argument vector and executes the learning
    attempts on the PUF instances.
    """
    if len(args) < 10 or len(args) > 11:
        stderr.write('LTF Array Simulator and Logistic Regression Learner\n')
        stderr.write('Usage:\n')
        stderr.write('sim_learn.py n k transformation combiner N restarts seed_instance seed_model [log_name]\n')
        stderr.write('               n: number of bits per Arbiter chain\n')
        stderr.write('               k: number of Arbiter chains\n')
        stderr.write('  transformation: used to transform input before it is used in LTFs\n')
        stderr.write('                  currently available:\n')
        stderr.write('                  - id  -- does nothing at all\n')
        stderr.write('                  - atf -- convert according to "natural" Arbiter chain\n')
        stderr.write('                           implementation\n')
        stderr.write('                  - mm  -- designed to achieve maximum PTF expansion length\n')
        stderr.write('                           only implemented for k=2 and even n\n')
        stderr.write('                  - lightweight_secure -- design by Majzoobi et al. 2008\n')
        stderr.write('                                          only implemented for even n\n')
        stderr.write('                  - shift_lightweight_secure -- design like Majzoobi\n')
        stderr.write('                                                et al. 2008, but with the shift\n')
        stderr.write('                                                operation executed first\n')
        stderr.write('                                                only implemented for even n\n')
        stderr.write('                  - soelter_lightweight_secure -- design like Majzoobi\n')
        stderr.write('                                                  et al. 2008, but one bit different\n')
        stderr.write('                                                  only implemented for even n\n')
        stderr.write('                  - 1_n_bent -- one LTF gets "bent" input, the others id\n')
        stderr.write('                  - 1_1_bent -- one bit gets "bent" input, the others id,\n')
        stderr.write('                                this is proven to have maximum PTF\n')
        stderr.write('                                length for the model\n')
        stderr.write('                  - polynomial -- challenges are interpreted as polynomials\n')
        stderr.write('                                  from GF(2^64). From the initial challenge c,\n')
        stderr.write('                                  the i-th Arbiter chain gets the coefficients \n')
        stderr.write('                                  of the polynomial c^(i+1) as challenge.\n')
        stderr.write('                                  For now only challenges with length n=64 are accepted.\n')
        stderr.write(
            '                  - permutation_atf -- for each Arbiter chain first a pseudorandom permutation \n')
        stderr.write('                                       is applied and thereafter the ATF transform.\n')
        stderr.write('                  - random -- Each Arbiter chain gets a random challenge derived from the\n')
        stderr.write('                              original challenge using a PRNG.\n')
        stderr.write('        combiner: used to combine the output bits to a single bit\n')
        stderr.write('                  currently available:\n')
        stderr.write('                  - xor     -- output the parity of all output bits\n')
        stderr.write('                  - ip_mod2 -- output the inner product mod 2 of all output\n')
        stderr.write('                               bits (even n only)\n')
        stderr.write('               N: number of challenge response pairs in the training set\n')
        stderr.write('        restarts: number of repeated initializations the learner\n')
        stderr.write('       instances: number of repeated initializations the instance\n')
        stderr.write('                  The number total learning attempts is restarts*instances.\n')
        stderr.write('   seed_instance: random seed used for LTF array instance\n')
        stderr.write('      seed_model: random seed used for the model in first learning attempt\n')
        stderr.write('      [log_name]: path to the logfile which contains results from all instances. The tool '
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
        stderr.write('Transformation %s unknown or currently not implemented\n' % transformation_name)
        quit()

    try:
        combiner = getattr(LTFArray, 'combiner_%s' % combiner_name)
    except AttributeError:
        stderr.write('Combiner %s unknown or currently not implemented\n' % combiner_name)
        quit()

    log_name = 'sim_learn'
    if len(args) == 11:
        log_name = args[10]

    stderr.write('Learning %s-bit %s XOR Arbiter PUF with %s CRPs and %s restarts.\n\n' % (n, k, N, restarts))
    stderr.write('Using\n')
    stderr.write('  transformation:       %s\n' % transformation)
    stderr.write('  combiner:             %s\n' % combiner)
    stderr.write('  instance random seed: 0x%x\n' % seed_instance)
    stderr.write('  model random seed:    0x%x\n' % seed_model)
    stderr.write('\n')

    experimenter = Experimenter(log_name)
    # create different experiment instances
    for j in range(instances):
        for start_number in range(restarts):
            l_name = '%s_%i_%i' % (log_name, j, start_number)
            experiment = ExperimentCorrelationAttack(
                log_name=l_name,
                n=n,
                k=k,
                N=N,
                seed_instance=seed_instance + j,
                seed_model=seed_model + j + start_number,
                seed_challenge_distance=0xbeef,
                seed_challenge=0xdead,
            )
            experimenter.queue(experiment)

    # run the instances
    experimenter.run()


if __name__ == '__main__':
    main(argv)
