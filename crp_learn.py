"""
This module is used for learning a PUF from known challenge-response pairs.
"""
import argparse
from pypuf import tools
from pypuf.learner.regression.logistic_regression import LogisticRegression
from pypuf.simulation.arbiter_based.ltfarray import LTFArray, CRCPuf


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
    parser.add_argument('-c', '--crc-puf', dest='crc_puf',
                        action='store_true', default=False,
                        help='use CRC-PUF transformation')
    parser.add_argument('-g', '--g_file', type=str,
                        help='file to read generator polynomials from (CRC-PUF only)')
    parser.add_argument('-m', type=uint, default=1,
                        help='response bits (CRC-PUF only)')
    args = parser.parse_args()

    if args.crc_puf and not args.g_file:
        raise argparse.ArgumentTypeError('g_file must be defined for CRC-PUFs')

    # read pairs from file
    trc, trr = tools.parse_file(args.file, args.n, args.m, 1, args.num_tr, args.in_11_notation)
    tec, ter = tools.parse_file(args.file, args.n, args.m, args.num_tr + 1, args.num_te,
                                args.in_11_notation)

    # CRC-PUF needs additional preprocessing
    if args.crc_puf:
        # read generator polynomials
        trg, _ = tools.parse_file(args.g_file, args.n, 0, 1, args.num_tr, args.in_11_notation)
        teg, _ = tools.parse_file(args.g_file, args.n, 0, args.num_tr + 1, args.num_te,
                                  args.in_11_notation)

        # apply the CRC-PUF transformation to the challenges
        training_set = tools.ChallengeResponseSet(
            CRCPuf.apply(trc, trg, args.m),
            trr.flatten()
        )
        testing_set = tools.ChallengeResponseSet(
            CRCPuf.apply(tec, teg, args.m),
            ter.flatten()
        )

    else:
        training_set = tools.ChallengeResponseSet(trc, trr.flatten())
        testing_set = tools.ChallengeResponseSet(tec, ter.flatten())

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
