"""This Module defines the experiments for the straight forward degree one approximation"""
import random
from collections import OrderedDict
from numpy import pi
from pypuf.experiments.experimenter import Experimenter
from pypuf.experiments.experiment.fourier_coefficient import ExperimentCFCA, ExperimentFCCRP
from pypuf.simulation.arbiter_based.ltfarray import LTFArray

parameter = {
    8: 2 ** 8,
    16: 2 ** 16,
    32: 10 ** 5,
    64: 10 ** 5
}

PATH = '/home/christoph/EXPERIMENTS/'
SAMPLE_COUNT = 922

def dictator(exp_parameter, experiment_fkt):
    """This function defines the experiments for the dictator controlled experiments."""
    instance_parameter = OrderedDict()
    instance_parameter['dictator'] = 1
    experiments = []
    for i in range(SAMPLE_COUNT):
        for n, challenge_count in exp_parameter.items():
            instance_parameter['n'] = n
            experiment = experiment_fkt(
                log_name=PATH + 'dictator_' + str(n) + str(i),
                challenge_count_min=1,
                challenge_count_max=challenge_count,
                challenge_seed= int(pi * (10 ** 3)) + 0xD1C2A204 + 540 + n + i,
                instance_gen=ExperimentFCCRP.create_dictator_instances,
                instance_parameter=instance_parameter
            )
            experiments.append(experiment)
    random.shuffle(experiments)
    experimenter = Experimenter(
        log_name=PATH + "controlled_experiment_dictator",
        experiments=experiments,
        status_display=True
    )
    experimenter.run()

def ip_mod2(exp_parameter, experiment_fkt):
    """This function defines the experiments for the ip_mod2 controlled experiments."""
    experiments = []
    instance_parameter = OrderedDict()
    for i in range(SAMPLE_COUNT):
        for n, challenge_count in exp_parameter.items():
            instance_parameter['n'] = n
            experiment = experiment_fkt(
                log_name=PATH + 'ip_mod2_' + str(n) + str(i),
                challenge_count_min=1,
                challenge_count_max=challenge_count,
                challenge_seed= int(pi * (10 ** 4)) + 0x1E92 + 1004 + n + i,
                instance_gen=ExperimentFCCRP.create_bent_instances,
                instance_parameter=instance_parameter
            )
            experiments.append(experiment)
    random.shuffle(experiments)
    experimenter = Experimenter(
        log_name=PATH + "controlled_experiment_ip_mod2",
        experiments=experiments,
        status_display=True
    )
    experimenter.run()

def ltf(exp_parameter, experiment_fkt):
    """This function defines the experiments for the ltf controlled experiments."""
    experiments = []
    instance_parameter = OrderedDict()
    instance_parameter['k'] = 1
    instance_parameter['transformation'] = LTFArray.transform
    instance_parameter['combiner'] = LTFArray.combiner_xor
    instance_parameter['bias'] = None
    instance_parameter['mu'] = 0
    instance_parameter['sigma'] = 1
    instance_parameter['weight_random_seed'] = int(pi * 10 ** 5)+ 0x1EFA44A9
    for i in range(SAMPLE_COUNT):
        for n, challenge_count in exp_parameter.items():
            instance_parameter['n'] = n
            experiment = experiment_fkt(
                log_name=PATH + 'ltf_' + str(n) + str(i),
                challenge_count_min=1,
                challenge_count_max=challenge_count,
                challenge_seed= int(pi * (10 ** 6)) + 0x1EF + 61 + n + i,
                instance_gen=ExperimentFCCRP.create_bent_instances,
                instance_parameter=instance_parameter
            )
            experiments.append(experiment)
    random.shuffle(experiments)
    experimenter = Experimenter(
        log_name=PATH + "controlled_experiment_ltf",
        experiments=experiments,
        status_display=True
    )
    experimenter.run()

dictator(parameter, ExperimentCFCA)
ip_mod2(parameter, ExperimentCFCA)
ltf(parameter, ExperimentCFCA)
