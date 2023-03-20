import numpy
from numpy import linalg


from comparison_matrix import ComparisonMatrix

random_index = [0, 0, 0.52, 0.89, 1.11, 1.25, 1.35, 1.4, 1.45, 1.49, 1.52]


# Compute normalized weights based on Geometric Mean method
def get_weight_geometric(matrix):
    column_sum = numpy.tile(numpy.sum(matrix, 0), [matrix.shape[0], 1])
    normalized_matrix = numpy.divide(matrix, column_sum)
    geometric_weight = numpy.prod(normalized_matrix, 1) ** (1.0 / matrix.shape[1])
    return geometric_weight / numpy.sum(geometric_weight)


# Compute normalized weights based on Eigen method - return both
# the maximum eigenvalue and the normalized weight vector
def get_weight_eigen(matrix):
    eigenvalues, normalized_eigenvectors = linalg.eig(matrix)
    max_eigenvalue_index = numpy.argmax(eigenvalues)
    # where max_eigenvector is the eigenvector corresponding to the maximum eigenvalue
    max_eigenvector = normalized_eigenvectors[:, max_eigenvalue_index]
    return eigenvalues[max_eigenvalue_index], max_eigenvector / numpy.sum(max_eigenvector)


# compute consistency index and consistency ratio based on
# outputs from getWeightEigen
def getCICR(eigL, eigV):
    # consistency index = (eigL- n) / (n-1)
    consistency_index = (eigL - eigV.shape[0]) / (eigV.shape[0] - 1)
    consistency_ratio = consistency_index / random_index[eigV.shape[0]]
    return consistency_index, consistency_ratio




comparison_matrices = {
    'PCBake': {
        'matrix': numpy.array(
            [[1, 1 / 4, 1 / 3, 1 / 2], [4, 1, 4 / 3, 2], [3, 3 / 4, 1, 6 / 5], [2, 1 / 2, 5 / 6, 1]]),
        'subcriteria': {
            'Size': {
                'matrix': numpy.array([[1, 5], [1 / 5, 1]]),
            },
            'Thermal Management': {
                'matrix': numpy.array([[1, 6], [1 / 6, 1]]),
            },
            'Accessibility': {
                'matrix': numpy.array(
                    [[1, 6 / 5], [5 / 6, 1]]),
            },
            'Cost': {
                'matrix': numpy.array(
                    [[1, 2], [1 / 2, 1]]),
            }
        }
    }
}


def calculate_weights(comparison_matrices, indent=0):
    for criteria in comparison_matrices:
        normalized_geometric = get_weight_geometric(comparison_matrices[criteria]['matrix'])
        max_eigenvalue, normalized_eigen = get_weight_eigen(comparison_matrices[criteria]['matrix'])
        consistency_index, consistency_ratio = getCICR(max_eigenvalue, normalized_eigen)

        comparison_matrices[criteria]['normalized_geometric'] = normalized_geometric
        comparison_matrices[criteria]['normalized_eigen'] = normalized_eigen
        comparison_matrices[criteria]['consistency_index'] = consistency_index
        comparison_matrices[criteria]['consistency_ratio'] = consistency_ratio

        print(
            f'{"  " * indent}{criteria} CI: {numpy.real_if_close(consistency_index):.2f}, CR: {numpy.real_if_close(consistency_ratio):.2f}')

        if 'subcriteria' in comparison_matrices[criteria]:
            calculate_weights(comparison_matrices[criteria]['subcriteria'], indent + 1)

        #


# returns ranking geometric, eigen
def calculate_ranking(comparison_matrix):
    if 'subcriteria' in comparison_matrix:
        geometric_rankings = []
        eigen_rankings = []
        for subcriteria in comparison_matrix['subcriteria']:
            geometric_ranking, eigen_ranking = calculate_ranking(comparison_matrix['subcriteria'][subcriteria])
            geometric_rankings.append(geometric_ranking)
            eigen_rankings.append(eigen_ranking)

        return numpy.matmul(numpy.stack(geometric_rankings, axis=-1),
                            comparison_matrix['normalized_geometric']), numpy.matmul(
            numpy.stack(eigen_rankings, axis=-1), comparison_matrix['normalized_eigen'])
    else:
        return comparison_matrix['normalized_geometric'], comparison_matrix['normalized_eigen']


if __name__ == "__main__":
    # (1) Calculate the Consistency Index (CI) and Consistency Ratios (CR) of each comparison matrix.
    print('\n######### 1) CI and CR')
    print('Calculating Consistency Index (CI) and Consistency Ratios (CR) for each comparison matrix')
    calculate_weights(comparison_matrices)

    # (2) Calculate the final rankings of the 4 cars using both the Geometric Mean method and the Eigen
    # method. Are the two rankings the same?
    print('\n######### 2) Geometric and Eigen Rankings')
    geometric_ranking, eigenvalue_ranking = calculate_ranking(comparison_matrices['PCBake'])
    print(f'Geometric rankings: {numpy.real_if_close(geometric_ranking)}')
    print(f'Eigenvalue rankings: {numpy.real_if_close(eigenvalue_ranking)}')

    # (3) A typographical mistake in data entry results in the following criteria level comparison matrix
    # (the red fonts highlight the mistake). How does the ranking change? Is it possible to spot this
    # mistake BEFORE computing the ranking?
    # print('\n######### 3) Typographic Mistake')
    # comparison_matrices['goal']['matrix'] = numpy.array(
    #     [[1, 5 / 7, 1 / 5, 5 / 9, 8], [7 / 5, 1, 5 / 9, 5 / 7, 4 / 5], [5, 9 / 5, 1, 7 / 3, 4 / 3],
    #      [9 / 5, 7 / 5, 3 / 7, 1, 9 / 7], [1 / 8, 5 / 4, 3 / 4, 7 / 9, 1]])
    # calculate_weights(comparison_matrices)
    # geometric_ranking, eigenvalue_ranking = calculate_ranking(comparison_matrices['goal'])
    # print(f'Geometric rankings: {numpy.real_if_close(geometric_ranking)}')
    # print(f'Eigenvalue rankings: {numpy.real_if_close(eigenvalue_ranking)}')
