"""This module defines the experiments for the Matules degree-one approximation."""
import random
from collections import OrderedDict
from numpy import pi
from pypuf.experiments.experimenter import Experimenter
from pypuf.experiments.experiment.fourier_coefficient import ExperimentCFCAMatules, ExperimentFCCRP
from pypuf.experiments.experiment.property_test import ExperimentPropertyTest
from pypuf.simulation.arbiter_based.ltfarray import LTFArray

PARAMETER = {
    64: 4 * 10 ** 5,
    128: 4 * 10 ** 5
}

PATH = '/home/christoph/EXPERIMENTS/'
SAMPLE_COUNT = 1116
MU = 0.05


def dictator(exp_parameter):
    """This function defines the experiments for the dictator matules controlled experiments."""
    experiments = []
    for i in range(923, SAMPLE_COUNT):
        for n, challenge_count in exp_parameter.items():
            instance_parameter = OrderedDict()
            instance_parameter['n'] = n
            instance_parameter['dictator'] = 1
            experiment = ExperimentCFCAMatules(
                log_name=PATH + 'matules_dictator_' + str(n) + str(i),
                challenge_count_min=1,
                challenge_count_max=challenge_count,
                challenge_seed=int(pi * (10 ** 3)) + 0xD1C2A204 + 540 + n + i,
                mu=MU,
                instance_gen=ExperimentFCCRP.create_dictator_instances,
                instance_parameter=instance_parameter
            )
            experiments.append(experiment)
    random.shuffle(experiments)
    experimenter = Experimenter(
        log_name=PATH + "controlled_experiment_dictator_matules923to1116",
        experiments=experiments,
        status_display=True,
    )
    experimenter.run()


def ip_mod2(exp_parameter):
    """This function defines the experiments for the ip_mod2 matules controlled experiments."""
    experiments = []
    for i in range(923, SAMPLE_COUNT):
        for n, challenge_count in exp_parameter.items():
            instance_parameter = OrderedDict()
            instance_parameter['n'] = n
            experiment = ExperimentCFCAMatules(
                log_name=PATH + 'matules_ip_mod2_' + str(n) + str(i),
                challenge_count_min=1,
                challenge_count_max=challenge_count,
                challenge_seed=int(pi * (10 ** 4)) + 0x1E92 + 1004 + n + i,
                mu=MU,
                instance_gen=ExperimentFCCRP.create_bent_instances,
                instance_parameter=instance_parameter
            )
            experiments.append(experiment)
    random.shuffle(experiments)
    experimenter = Experimenter(
        log_name=PATH + "controlled_experiment_ip_mod2_matules923to1116",
        experiments=experiments,
        status_display=True,
    )
    experimenter.run()


def ltf(exp_parameter):
    """This function defines the experiments for the ltf matules controlled experiments."""
    experiments = []
    for i in range(923, SAMPLE_COUNT):
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
            experiment = ExperimentCFCAMatules(
                log_name=PATH + 'matules_ltf_' + str(n) + str(i),
                challenge_count_min=1,
                challenge_count_max=challenge_count,
                challenge_seed=int(pi * (10 ** 6)) + 0x1EF + 61 + n + i,
                mu=MU,
                instance_gen=ExperimentPropertyTest.create_ltf_arrays,
                instance_parameter=instance_parameter
            )
            experiments.append(experiment)
    random.shuffle(experiments)
    experimenter = Experimenter(
        log_name=PATH + "controlled_experiment_ltf_matules923to1116",
        experiments=experiments,
        status_display=True,
    )
    experimenter.run()
print('calculate dictator matules')
dictator(PARAMETER)
print('\ncalculate ip_mod2 matules')
ip_mod2(PARAMETER)
print('\ncalculate ltf matules')
ltf(PARAMETER)