from numpy import random, amin, amax, mean, array, append
from pypuf import simulation, learner, tools
import time
from sys import argv, stdout, stderr

if len(argv) != 9:
    stderr.write('LTF Array Simulator and Logistic Regression Learner\n')
    stderr.write('Usage:\n')
    stderr.write('sim_learn.py n k transformation combiner N restarts seed_ltf seed_model\n')
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
    stderr.write('        combiner: used to combine the output bits to a single bit\n')
    stderr.write('                  currently available:\n')
    stderr.write('                  - xor -- output the parity of all output bits\n')
    stderr.write('               N: number of challenge response pairs in the training set\n')
    stderr.write('        restarts: number of repeated initializations the learner\n')
    stderr.write('        seed_ltf: random seed used for LTF array instance\n')
    stderr.write('      seed_model: random seed used for the model in first learning attempt\n')
    quit(1)

n = int(argv[1])
k = int(argv[2])
transformation_name = argv[3]
combiner_name = argv[4]
N = int(argv[5])
restarts = int(argv[6])
seed_ltf = int(argv[7], 16)
seed_model = int(argv[8], 16)

random.seed(seed_ltf) # reproduce 'random' numbers

transformation = None
combiner = None

try:
    transformation = getattr(simulation.LTFArray, 'transform_%s' % transformation_name)
except AttributeError:
    stderr.write('Transformation %s unknown or currently not implemented\n' % transformation_name)
    quit()

try:
    combiner = getattr(simulation.LTFArray, 'combiner_%s' % combiner_name)
except AttributeError:
    stderr.write('Combiner %s unknown or currently not implemented\n' % combiner_name)
    quit()

stderr.write('Learning %s-bit %s XOR Arbiter PUF with %s CRPs and %s restarts.\n\n' % (n, k, N, restarts))
stderr.write('Using\n')
stderr.write('  transformation:    %s\n' % transformation)
stderr.write('  combiner:          %s\n' % combiner)
stderr.write('  ltf random seed:   0x%x\n' % seed_ltf)
stderr.write('  model random seed: 0x%x\n' % seed_model)
stderr.write('\n')

instance = simulation.LTFArray(
    weight_array=simulation.LTFArray.normal_weights(n, k),
    transform=transformation,
    combiner=combiner,
)

lr_learner = learner.LogisticRegression(
    tools.TrainingSet(instance=instance, N=N),
    n,
    k,
    transformation=transformation,
    combiner=combiner,
)

accuracy = array([])
training_times = array([])
iterations = array([])

random.seed(seed_model)

for i in range(restarts):
    stderr.write('\r%5i/%5i         ' % (i+1, restarts))
    start = time.time()
    model = lr_learner.learn()
    end = time.time()
    training_times = append(training_times, end - start)
    dist = tools.approx_dist(instance, model, min(1000, 2 ** n))
    accuracy = append(accuracy, 1 - dist)
    iterations = append(iterations, lr_learner.iteration_count)
    # output test result in machine-friendly format
    # seed_ltf seed_model idx_restart n k N transformation combiner iteration_count time accuracy
    stdout.write(' '.join(
        [
            '0x%x' % seed_ltf,
            '0x%x' % seed_model,
            '%5d' % i,
            '%3d' % n,
            '%2d' % k,
            '%6d' % N,
            '%s' % transformation_name,
            '%s' % combiner_name,
            '%4d' % lr_learner.iteration_count,
            '%9.3f' % (end - start),
            '%1.5f' % (1 - dist),
        ]
    ) + '\n')
    #stderr.write('training time:                % 5.3fs' % (end - start))
    #stderr.write('min training distance:        % 5.3f' % lr_learner.min_distance)
    #stderr.write('test distance (1000 samples): % 5.3f\n' % dist)

stderr.write('\r              \r')
stderr.write('\n\n')
stderr.write('training times: %s\n' % training_times)
stderr.write('iterations: %s\n' % iterations)
stderr.write('test accuracy: %s\n' % accuracy)
stderr.write('\n\n')
stderr.write('min/avg/max training time  : % 9.3fs /% 9.3fs /% 9.3fs\n' % (amin(training_times), mean(training_times), amax(training_times)))
stderr.write('min/avg/max iteration count: % 9.3f  /% 9.3f  /% 9.3f \n' % (amin(iterations), mean(iterations), amax(iterations)))
stderr.write('min/avg/max test accuracy  : % 9.3f  /% 9.3f  /% 9.3f \n' % (amin(accuracy), mean(accuracy), amax(accuracy)))
stderr.write('\n\n')
