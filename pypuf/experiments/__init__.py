"""
Disable numpy's multiprocessing as soon as possible when importing the
Experimenter. Note that this code will raise an exception if numpy was
already imported, as deactivating multiprocessing is then impossible.
"""
from pypuf.experiments.experimenter import Experimenter


Experimenter.disable_auto_multiprocessing(raise_exception=False)
