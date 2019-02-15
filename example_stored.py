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
    n, k, N, filename = 128, 1, 4, 'test/data/stub-crps.txt'

    # create a simulation with random (Gaussian) weights
    # for 64-bit 2-XOR
    instance = LTFArray(
        weight_array=LTFArray.normal_weights(n=n, k=k),
        transform=LTFArray.transform_atf,
        combiner=LTFArray.combiner_xor,
    )

    # create the learner
    lr_learner = LogisticRegression(
        t_set=tools.TrainingSet.from_file(instance, N, filename),
        n=n,
        k=k,
        transformation=LTFArray.transform_atf,
        combiner=LTFArray.combiner_xor,
    )

    # learn and test the model
    model = lr_learner.learn()
    accuracy = 1 - tools.approx_dist(instance, model, 2000)

    # output the result
    print('Learned a {}-bit {}-xor XOR Arbiter PUF from {} CRPs with accuracy {}'.format(n, k, N, accuracy))


if __name__ == '__main__':
    main()
