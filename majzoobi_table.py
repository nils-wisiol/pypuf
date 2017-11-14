from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf import tools
from numpy import array, sum, zeros, ones, round, roll, mean

transform = LTFArray.transform_lightweight_secure_original


def gen_table(n, k, samples):
    inputs = array(list(tools.sample_inputs(n, samples)))
    matrix_rot = zeros((k, k))
    matrix_rot2 = zeros((k, k))
    matrix_cor = ones((k, k))
    transformed = transform(inputs, k)
    transformed = tools.append_last(transformed, 1)

    for source_puf in range(k):
        source_inputs = transformed[:, source_puf, :]
        for target_puf in range(source_puf + 1, k):
            if source_puf == target_puf:
                continue
            target_inputs = transformed[:, target_puf, :]
            max_corr = 0
            best_rot = 0
            best_rot2 = 0
            n = len(target_inputs[0])
            for rotation_source in range(n):
                rotation_target = n - rotation_source
                rotated_source = roll(source_inputs, rotation_source, axis=1)
                rotated_target = roll(target_inputs, rotation_target, axis=1)

                prod_original = array([source_inputs[:, i] * target_inputs[:, j] for i in range(n) for j in range(n)])
                prod_shifted = array([rotated_target[:, i] * rotated_source[:, j] for i in range(n) for j in range(n)])
                correlation = 0.5 + 0.5 * sum(prod_original * prod_shifted, axis=1) / samples

                correlation_mean = mean(correlation)

                if correlation_mean > max_corr:
                    max_corr = correlation_mean
                    best_rot = rotation_source
                    best_rot2 = rotation_target
            matrix_rot[source_puf][target_puf] = best_rot
            matrix_cor[source_puf][target_puf] = max_corr
            matrix_rot2[source_puf][target_puf] = best_rot2

    for i in range(k):
        for j in range(i):
            matrix_rot[i, j] = matrix_rot2[j, i]
            matrix_cor[i, j] = matrix_cor[j, i]
    return matrix_rot, matrix_cor


def main():
    n = 64
    k = 6
    matrix_rot, matrix_cor = gen_table(n, k, 1000)
    print(matrix_rot)
    print(round(matrix_cor, 2))


main()
