from numpy.random import RandomState
from sys import argv, stderr
import json

from pypuf.experiments.experiment.input_transform_experiment import InputTransformExperiment
from pypuf.experiments.experimenter import Experimenter
from pypuf.simulation.arbiter_based.ltfarray import LTFArray


def main(args):
    """This method includes the main functionality of the module it parses the argument vector
    and executes the learning attempts on the PUF instances.
    """
    if len(args) < 5 or len(args) > 9:
        stderr.write('\n***Input Transform Comparison***\n\n')
        stderr.write('Usage:\n')
        stderr.write(
            'python exec_input_transform_comparison.py k n start end step repeat [suffix]\n')
        stderr.write('  k:          number of Arbiter chains\n')
        stderr.write('  n:          number of bits per Arbiter chain\n')
        stderr.write('  start:      lowest number of different challenges in the training set\n')
        stderr.write('  end:        highest number of different challenges in the training set\n')
        stderr.write('  step:       step size between numbers of different challenges in the training set\n')
        stderr.write('  [repeat]:     number of repetitions of every experiment\n')
        stderr.write('  [suffix]:   suffix for the name of the logfile.\n')
        stderr.write('  [seed_i]:   random seed for creating LTF array instance and simulating its noise\n')
        stderr.write('  [seed_c]:   random seed for sampling challenges\n')
        quit(1)

    # Use obligatory parameters
    k = json.loads(args[1])
    n = json.loads(args[2])
    start = json.loads(args[3])
    end = json.loads(args[4])
    step = json.loads(args[5])

    # Use or initialize optional parameters
    repeat = 1
    suffix = ''
    seed_i = RandomState().randint(2**32)
    seed_c = RandomState().randint(2**32)
    if len(args) >= 7:
        repeat = int(args[6])
        if len(args) >= 8:
            suffix = str(args[7])
            if len(args) >= 9:
                seed_i = int(args[8])
                if len(args) >= 10:
                    seed_c = int(args[9])

    stderr.write('Following parameters are used:\n')
    stderr.write(str(argv[1:]))
    stderr.write('\nThe following seeds are used for generating pseudo random numbers.\n')
    stderr.write('  seed for instance:      0x%x\n' % seed_i)
    stderr.write('  seed for challenges:    0x%x\n' % seed_c)
    stderr.write('\n')

    experiments = list()

    log_name = 'trafos_k%i_n%i' % (k, n) + suffix
    params = list()
    for i in range(start, end, step):
        params.append([k, n, LTFArray.transform_id, LTFArray.combiner_xor, i])
        params.append([k, n, LTFArray.transform_atf, LTFArray.combiner_xor, i])
        params.append([k, n, LTFArray.transform_aes_substitution, LTFArray.combiner_xor, i])
        params.append([k, n, LTFArray.transform_lightweight_secure, LTFArray.combiner_xor, i])
        if n == 64:
            params.append([k, n, LTFArray.transform_fixed_permutation, LTFArray.combiner_xor, i])
        params.append([k, n, LTFArray.transform_random, LTFArray.combiner_xor, i])

    for i in range(len(params)):
        for j in range(repeat):
            experiment = InputTransformExperiment(
                log_name=log_name + '_%i' % i,
                k=params[i][0],
                n=params[i][1],
                transform=params[i][2],
                combiner=params[i][3],
                num=params[i][4],
                seed_instance=RandomState((seed_i + i*repeat + j) % 2**32).randint(2**32),
                seed_challenges=RandomState((seed_c + i*repeat + j) % 2**32).randint(2**32)
            )
            experiments.append(experiment)

    experimenter = Experimenter(result_log_name=log_name, cpu_limit=1)
    for e in experiments:
        experiment_id = experimenter.queue(e)
    experimenter.run()


if __name__ == '__main__':
    main(argv)
