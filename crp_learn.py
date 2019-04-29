"""
This module is used for learning a PUF from known challenge-response pairs.
"""
import argparse
from pypuf import tools
from pypuf.learner.regression.logistic_regression import LogisticRegression
from pypuf.simulation.arbiter_based.ltfarray import LTFArray


def uint(val):
    """
    Assures that the passed integer is positive.
    """
    ival = int(val)
    if ival <= 0:
        raise argparse.ArgumentTypeError('{} is not a positive integer'.format(val))
    return ival


def main():
    """
    Learns and evaluates a PUF.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('n', type=uint,
                        help='challenge bits')
    parser.add_argument('k', type=uint,
                        help='number of arbiter chains')
    parser.add_argument('num_tr', type=uint,
                        help='number of CRPs to use for training')
    parser.add_argument('num_te', type=uint,
                        help='number of CRPs to use for testing')
    parser.add_argument('file', type=str,
                        help='file to read CRPs from')
    parser.add_argument('-1', '--11-notation', dest='in_11_notation',
                        action='store_true', default=False,
                        help='file is in -1,1 notation (default is 0,1)')
    args = parser.parse_args()

    # read pairs from file
    training_set = tools.parse_file(args.file, args.n, 1, args.num_tr,
                                    args.in_11_notation)
    testing_set = tools.parse_file(args.file, args.n, args.num_tr + 1,
                                   args.num_te, args.in_11_notation)

    # create the learner
    lr_learner = LogisticRegression(
        t_set=training_set,
        n=args.n,
        k=args.k,
        transformation=LTFArray.transform_atf,
        combiner=LTFArray.combiner_xor,
    )

    # learn and test the model
    model = lr_learner.learn()
    accuracy = 1 - tools.approx_dist_nonrandom(model, testing_set)

    # output the result
    print('Learned a {}-bit {}-xor XOR Arbiter PUF from {} CRPs with accuracy {}'
          .format(args.n, args.k, args.num_tr, accuracy))


if __name__ == '__main__':
    main()
