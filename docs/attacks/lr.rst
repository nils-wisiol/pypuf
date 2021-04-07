PUF Modeling Attacks
====================

One promise of Physically Unclonable Functions is an increase in hardware security, compared to non-volatile memory key
storage solutions. Consequently, the attacker model for PUFs is broadly defined. Usually, the attacker gets physical
access to the PUF token for a longer but limited amount of time [WMPN19]_.

Within this attacker model, it is possible to subject a PUF to *modeling attacks*, where the attacker tries to
extrapolate the PUF token's behavior, i.e. to predict the behavior of the PUF when stimulated with unseen challenges.
If an attacker is able to predict the PUF token's behavior, a remote party cannot distinguish between the PUF token and
the predictive model, breaking the security of the PUF.

Currently implemented attacks in pypuf are all *offline*, i.e. they run on pre-recorded information (as opposed to
having online access for adaptively chosen queries).

.. autoclass:: pypuf.attack.base.OfflineAttack
    :members: __init__, fit, model

