"""
This module is only used to show some example usage of the framework.
"""
from pypuf import tools
from pypuf.learner.regression.logistic_regression import LogisticRegression
from pypuf.simulation.arbiter_based.ltfarray import LTFArray


def main():
    """
    Run an example how to use pypuf.
    Developers Notice: Changes here need to be mirrored to README!
    """

    # create a simulation with random (Gaussian) weights
    # for 64-bit 2-XOR
    instance = LTFArray(
        weight_array=LTFArray.normal_weights(n=64, k=2),
        transform=LTFArray.transform_atf,
        combiner=LTFArray.combiner_xor,
    )

    # create the learner
    lr_learner = LogisticRegression(
        t_set=tools.TrainingSet(instance=instance, N=12000),
        n=64,
        k=2,
        transformation=LTFArray.transform_atf,
        combiner=LTFArray.combiner_xor,
    )

    # learn and test the model
    model = lr_learner.learn()
    accuracy = 1 - tools.approx_dist(instance, model, 10000)

    # output the result
    print('Learned a 64bit 2-xor XOR Arbiter PUF from 12000 CRPs with accuracy %f' % accuracy)


if __name__ == '__main__':
    main()
