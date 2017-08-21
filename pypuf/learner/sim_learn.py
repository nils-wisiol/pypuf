from numpy import amin, amax, mean, array, append
from numpy.random import RandomState
from pypuf.learner.regression.logistic_regression import LogisticRegression
from pypuf.learner.liu.polytope_algorithm import PolytopeAlgorithm
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf import tools
import time
from sys import argv, stdout, stderr

if len(argv) != 11:
    stderr.write('LTF Array Simulator and Learner (Logistic Regression, Polytope)\n')
    stderr.write('Usage:\n')
    stderr.write('sim_learn.py n k transformation combiner N restarts seed_instance seed_model\n')
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
    stderr.write('                  - 1_n_bent -- one LTF gets "bent" input, the others id\n')
    stderr.write('                  - 1_1_bent -- one bit gets "bent" input, the others id,\n')
    stderr.write('                                this is proven to have maximum PTF\n')
    stderr.write('                                length for the model\n')
    stderr.write('        combiner: used to combine the output bits to a single bit\n')
    stderr.write('                  currently available:\n')
    stderr.write('                  - xor     -- output the parity of all output bits\n')
    stderr.write('                  - ip_mod2 -- output the inner product mod 2 of all output\n')
    stderr.write('                               bits (even n only)\n')
    stderr.write('               N: number of challenge response pairs in the training set\n')
    stderr.write('        restarts: number of repeated initializations the learner\n')
    stderr.write('                  use float number x, 0<x<1 to repeat until given accuracy\n')
    stderr.write('       instances: number of repeated initializations the instance\n')
    stderr.write('                  The number total learning attempts is restarts*instances.\n')
    stderr.write('   seed_instance: random seed used for LTF array instance\n')
    stderr.write('      seed_model: random seed used for the model in first learning attempt\n')
    stderr.write('         learner: chosen learning algorithm given as Integer \n')
    stderr.write('                  currently available:\n')
    stderr.write('                  - 1      -- Logistic Regression\n')
    stderr.write('                  - 2      -- Polytope\n')
    quit(1)

n = int(argv[1])
k = int(argv[2])
transformation_name = argv[3]
combiner_name = argv[4]
N = int(argv[5])

if float(argv[6]) < 1:
    restarts = float('inf')
    convergence = float(argv[6])
else:
    restarts = int(argv[6])
    convergence = 1.1

instances = int(argv[7])

seed_instance = int(argv[8], 16)
seed_model = int(argv[9], 16)

algorithm = int(argv[10])

# reproduce 'random' numbers and avoid interference with other random numbers drawn
instance_prng = RandomState(seed=seed_instance)
model_prng = RandomState(seed=seed_model)

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

stderr.write('Learning %s-bit %s XOR Arbiter PUF with %s CRPs and %s restarts.\n\n' % (n, k, N, restarts))
stderr.write('Using\n')

if algorithm==1:
    stderr.write('  Logistic Regression \n')
elif algorithm==2:
    stderr.write('  Polytope Algorithm \n')
else: 
    stderr.write('  Invalid Input: using Logistic Regression as default \n')
    
stderr.write('  transformation:       %s\n' % transformation)
stderr.write('  combiner:             %s\n' % combiner)
stderr.write('  instance random seed: 0x%x\n' % seed_instance)
stderr.write('  model random seed:    0x%x\n' % seed_model)
stderr.write('\n')

accuracy = array([])
training_times = array([])
iterations = array([])

for j in range(instances):

    stderr.write('----------- Choosing new instance. ---------\n')

    instance = LTFArray(
        weight_array=LTFArray.normal_weights(n, k, random_instance=instance_prng),
        transform=transformation,
        combiner=combiner,
    )
    
    if algorithm==2:
        learner = PolytopeAlgorithm(
                instance,
                tools.TrainingSet(instance=instance, N=N),
                n,
                k,
                transformation=transformation,
                combiner=combiner,
                weights_prng=model_prng,
                )
    else:
        learner = LogisticRegression(
                tools.TrainingSet(instance=instance, N=N),
                n,
                k,
                transformation=transformation,
                combiner=combiner,
                weights_prng=model_prng,
                )

    i = 0
    dist = 1

    while i < restarts and 1 - dist < convergence:
        stderr.write('\r%5i/%5i         ' % (i+1, restarts if restarts < float('inf') else 0))
        start = time.time()
        model = learner.learn()
        end = time.time()
        training_times = append(training_times, end - start)
        dist = tools.approx_dist(instance, model, min(10000, 2 ** n))
        accuracy = append(accuracy, 1 - dist)
        iterations = append(iterations, learner.iteration_count)
        # output test result in machine-friendly format
        # seed_ltf seed_model idx_restart n k N transformation combiner iteration_count time accuracy
        stdout.write(' '.join(
            [
                '0x%x' % seed_instance,
                '0x%x' % seed_model,
                '%5d' % i,
                '%3d' % n,
                '%2d' % k,
                '%6d' % N,
                '%s' % transformation_name,
                '%s' % combiner_name,
                '%4d' % learner.iteration_count,
                '%9.3f' % (end - start),
                '%1.5f' % (1 - dist),
            ]
        ) + '\n')
        #stderr.write('training time:                % 5.3fs' % (end - start))
        #stderr.write('min training distance:        % 5.3f' % lr_learner.min_distance)
        #stderr.write('test distance (1000 samples): % 5.3f\n' % dist)
        i += 1

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
