"""
Modeling Attacks on Physically Unclonable Functions (PUFs).
"""
from abc import ABC
from typing import Optional

from ..io import ChallengeInformationSet
from ..simulation.base import Simulation


class Attack:
    """
    PUF Modeling attack.
    """

    def __init__(self) -> None:
        """Initialize the attack."""
        super().__init__()
        self._model: Optional[Simulation] = None

    def fit(self) -> Simulation:
        """
        Runs the attack configured in this attack object. The obtained model is stored as :meth:`model` property and
        provided as return value.
        """
        raise NotImplementedError

    @property
    def model(self) -> Optional[Simulation]:
        """
        The model that was obtained running the :meth:`fit` method, ``None`` if :meth:`fit` was not run yet.
        """
        return self._model


class OnlineAttack(Attack, ABC):
    """
    PUF Modeling attack with online (i.e., adaptive) query access to the PUF under attack.
    """

    def __init__(self, target: Simulation) -> None:
        super().__init__()
        self.target = target


class OfflineAttack(Attack, ABC):
    """
    A modeling attack based on a prerecorded information about a PUF token's behavior on given challenges.
    """

    def __init__(self, crps: ChallengeInformationSet) -> None:
        """
        Initialize the modeling attack. After initialization, the attack can be run using the :meth:`fit` function.

        :param crps: Information about observed behavior of the PUF on known challenges.
        """
        super().__init__()
        self.crps = crps
