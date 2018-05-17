"""This Module defines the experiments for the straight forward degree one approximation"""
import random
from collections import OrderedDict
from numpy import pi
from pypuf.experiments.experimenter import Experimenter
from pypuf.experiments.experiment.fourier_coefficient import ExperimentCFCA, ExperimentFCCRP
from pypuf.experiments.experiment.property_test import ExperimentPropertyTest
from pypuf.simulation.arbiter_based.ltfarray import LTFArray

PARAMETER = {
    64: 10 ** 5,
    128: 10 ** 5
}

PATH = '/media/chris/Expansion Drive/EXPERIMENTS/'
SAMPLE_COUNT = 1000


def dictator(exp_parameter, experiment_fkt):
    """This function defines the experiments for the dictator controlled experiments."""
    experiments = []
    for i in range(SAMPLE_COUNT):
        for n, challenge_count in exp_parameter.items():
            instance_parameter = OrderedDict()
            instance_parameter['n'] = n
            instance_parameter['dictator'] = 1
            experiment = experiment_fkt(
                log_name=PATH + 'dictator_' + str(n) + str(i),
                challenge_count_min=1,
                challenge_count_max=challenge_count,
                challenge_seed=int(pi * (10 ** 3)) + 0xD1C2A204 + 540 + n + i,
                instance_gen=ExperimentFCCRP.create_dictator_instances,
                instance_parameter=instance_parameter
            )
            experiments.append(experiment)
    random.shuffle(experiments)
    experimenter = Experimenter(
        log_name=PATH + "controlled_experiment_dictator",
        experiments=experiments,
        status_display=True,
    )
    experimenter.run()


def ip_mod2(exp_parameter, experiment_fkt):
    """This function defines the experiments for the ip_mod2 controlled experiments."""
    experiments = []
    for i in range(SAMPLE_COUNT):
        for n, challenge_count in exp_parameter.items():
            instance_parameter = OrderedDict()
            instance_parameter['n'] = n
            experiment = experiment_fkt(
                log_name=PATH + 'ip_mod2_' + str(n) + str(i),
                challenge_count_min=1,
                challenge_count_max=challenge_count,
                challenge_seed=int(pi * (10 ** 4)) + 0x1E92 + 1004 + n + i,
                instance_gen=ExperimentFCCRP.create_bent_instances,
                instance_parameter=instance_parameter
            )
            experiments.append(experiment)
    random.shuffle(experiments)
    experimenter = Experimenter(
        log_name=PATH + "controlled_experiment_ip_mod2",
        experiments=experiments,
        status_display=True,
    )
    experimenter.run()


def ltf(exp_parameter, experiment_fkt):
    """This function defines the experiments for the ltf controlled experiments."""
    experiments = []
    for i in range(SAMPLE_COUNT):
        for n, challenge_count in exp_parameter.items():
            instance_parameter = OrderedDict()
            instance_parameter['n'] = n
            instance_parameter['k'] = 1
            instance_parameter['transformation'] = LTFArray.transform_id
            instance_parameter['combiner'] = LTFArray.combiner_xor
            instance_parameter['bias'] = None
            instance_parameter['mu'] = 0
            instance_parameter['sigma'] = 1
            instance_parameter['weight_random_seed'] = int(pi * 10 ** 5) + 0x1EFA44A9
            experiment = experiment_fkt(
                log_name=PATH + 'ltf_' + str(n) + str(i),
                challenge_count_min=1,
                challenge_count_max=challenge_count,
                challenge_seed=int(pi * (10 ** 6)) + 0x1EF + 61 + n + i,
                instance_gen=ExperimentPropertyTest.create_ltf_arrays,
                instance_parameter=instance_parameter
            )
            experiments.append(experiment)
    random.shuffle(experiments)
    experimenter = Experimenter(
        log_name=PATH + "controlled_experiment_ltf",
        experiments=experiments,
        status_display=True,
    )
    experimenter.run()
print('calculate dictator')
dictator(PARAMETER, ExperimentCFCA)
print('\ncalculate ip_mod2')
ip_mod2(PARAMETER, ExperimentCFCA)
print('\ncalculate ltf')
ltf(PARAMETER, ExperimentCFCA)
