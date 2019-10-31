"""
This module is a command line tool which provides an interface for experiments which are designed to learn an arbiter
PUF LTFarray simulation with the reliability based CMAES learning algorithm. If you want to use this tool you will have
to define nine parameters which define the experiment.
"""
from sys import argv, stderr
from pypuf.experiments.experimenter import Experimenter
from matplotlib.pyplot import figure
import seaborn as sns
import numpy as np

import numpy.random as rnd

from pypuf.experiments.experiment.reliability_based_cmaes import ExperimentReliabilityBasedCMAES, Parameters



def main(args):
    """This method includes the main functionality of the module it parses the argument vector
    and executes the learning attempts on the PUF instances.
    """
    if len(args) < 9 or len(args) > 15:
        stderr.write('\n***LTF Array Simulator and Reliability based CMAES Learner***\n\n')
        stderr.write('Usage:\n')
        stderr.write(
            'python sim_rel_cmaes_attack.py n k noisiness num reps pop_size [seed_i] [seed_c] [seed_m] [result_log_name]\n')
        stderr.write('  n:          number of bits per Arbiter chain\n')
        stderr.write('  k:          number of Arbiter chains\n')
        stderr.write('  noisiness:  proportion of noise scale related to the scale of variability\n')
        stderr.write('  num:        number of different challenges in the training set\n')
        stderr.write('  reps:       number of responses for each challenge in the training set\n')
        stderr.write('  pop_size:   number of solution points sampled per iteration within the CMAES algorithm\n')
        stderr.write('  limit_s:    max number of iterations with consistent fitness within the CMAES algorithm\n')
        stderr.write('  limit_i:    max number of overall iterations within the CMAES algorithm\n')
        stderr.write('  [result_log_name]:     name of the logfile which contains all results from the experiment.\n'
                     '                   The tool will add a ".log" to result_log_name. The default is "sim_rel_cmaes.log"\n')
        stderr.write('  [instances]:    number of repeated initializations of the instance\n')
        stderr.write('  [attempts]:     number of repeated initializations of the learner for the same instance\n')
        stderr.write('                   The number of total learning executions is instances times attempts.\n')
        stderr.write('  [seed_i]:       random seed for creating LTF array instance and simulating its noise\n')
        stderr.write('  [seed_c]:       random seed for sampling challenges\n')
        stderr.write('  [seed_m]:       random seed for modelling LTF array instance\n')
        quit(1)

    # Use obligatory parameters
    n = int(args[1])
    k = int(args[2])
    noisiness = float(args[3])
    num = int(args[4])
    reps = int(args[5])
    pop_size = int(args[6])
    limit_s = int(args[7])
    limit_i = int(args[8])

    # Initialize or use optional parameters
    result_log_name = 'sim_rel_cmaes'
    instances = 1
    attempts = 1
    seed_i = rnd.randint(0, 2 ** 32)
    seed_c = rnd.randint(0, 2 ** 32)
    seed_m = rnd.randint(0, 2 ** 32)
    if len(args) >= 10:
        result_log_name = args[9]
        if len(args) >= 11:
            instances = int(args[10])
            if len(args) >= 12:
                attempts = int(args[11])
                if len(args) >= 13:
                    seed_i = int(args[12], 0)
                    if len(args) >= 14:
                        seed_c = int(args[13], 0)
                        if len(args) == 15:
                            seed_m = int(args[14], 0)

    stderr.write('Learning %i time(s) each %i (%i,%i)-XOR Arbiter PUF(s) with %f noisiness, '
                 'using %i different %i times repeated CRPs.\n'
                 'There, %i solution points are sampled each iteration of the CMAES algorithm. '
                 'Among other termination criteria, it stops if the fitness stagnates for %i iterations '
                 'or the total number of iterations equals %i.\n'
                 % (attempts, instances, n, k, noisiness, num, reps, pop_size, limit_s, limit_i))
    stderr.write('The following seeds are used for generating pseudo random numbers.\n')
    stderr.write('  seed for instance:      0x%x\n' % seed_i)
    stderr.write('  seed for challenges:    0x%x\n' % seed_c)
    stderr.write('  seed for model:         0x%x\n' % seed_m)
    stderr.write('\n')

    # Create different experiment instances
    experiments = []
    for instance in range(instances):
        for attempt in range(attempts):
            l_name = result_log_name
            if instances > 1 or attempts > 1:
                l_name += '_%i_%i' % (instance, attempt)
            experiment = ExperimentReliabilityBasedCMAES(
                progress_log_name=l_name,
                parameters=Parameters(
                    seed_instance=(seed_i + instance) % 2 ** 32,
                    seed_challenges=(seed_c + instance) % 2 ** 32,
                    k=k,
                    n=n,
                    transform='id',
                    combiner='xor',
                    noisiness=noisiness,
                    num=num,
                    reps=reps,
                    seed_model=(seed_m + attempt) % 2 ** 32,
                    pop_size=pop_size,
                    limit_stag=limit_s,
                    limit_iter=limit_i,
                )
            )
            experiments.append(experiment)

    experimenter = Experimenter(result_log_name, results_file=result_log_name + '.csv')
    for e in experiments:
        experimenter.queue(e)
    # Run the instances
    experimenter.run()

    # Prepare data
    data = experimenter.results
    accuracy = data['accuracy']
    time = data['measured_time']

    # Skip analysis if only one sample
    if len(accuracy) <= 1:
        print(accuracy[0])
        print(time[0])
        return

    # Plot
    fig = figure()
    sns.set()
    ax = sns.distplot(accuracy)
    ax.set_xlabel('Accuracy')
    fig.suptitle('CMAES Accuracy Distribution, k={}, n={}'.format(k, n))
    fig.savefig('figures/cmaes_accuracy_dist', bbox_inches='tight', pad_inches=.5)

    print("mean accuracy = " + str(np.mean(accuracy)))
    print("mean time = " + str(np.mean(time)))
    print("% success = " + str(sum(accuracy >= .95)/len(accuracy)))
    print("% success >90% = " + str(sum(accuracy >= .90)/len(accuracy)))


if __name__ == '__main__':
    main(argv)
