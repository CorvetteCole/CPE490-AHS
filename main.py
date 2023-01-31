import numpy as np
from numpy import linalg

random_index = [0, 0, 0.52, 0.89, 1.11, 1.25, 1.35, 1.4, 1.45, 1.49, 1.52]


# Compute normalized weights based on Geometric Mean method
def getWeightGM(pwMat):
    colsum = np.tile(np.sum(pwMat, 0), [pwMat.shape[0], 1])
    nmat = np.divide(pwMat, colsum)
    gmWeight = np.prod(nmat, 1) ** (1.0 / pwMat.shape[1])
    return gmWeight / np.sum(gmWeight)


# Compute normalized weights based on Eigen method - return both
# the maximum eigenvalue and the normalized weight vector
def getWeightEigen(pwMat):
    eigL, eigV = linalg.eig(pwMat)
    maxEigL = np.max(eigL)
    maxEigLIndex = np.argmax(eigL)
    maxEigV = eigV[:, maxEigLIndex]
    return maxEigL, maxEigV / np.sum(maxEigV)


# compute consistency index and consistency ratio based on
# outputs from getWeightEigen
def getCICR(eigL, eigV):
    # consistency index = (eigL- n) / (n-1)
    consistency_index = (eigL - eigV.shape[0]) / (eigV.shape[0] - 1)
    consistency_ratio = consistency_index / random_index[eigV.shape[0]]
    return consistency_index, consistency_ratio


comparison_matrices = {
    'goal': {
        'matrix': np.array(
            [[1, 5 / 7, 1 / 5, 5 / 9, 1 / 8], [7 / 5, 1, 5 / 9, 5 / 7, 4 / 5], [5, 9 / 5, 1, 7 / 3, 4 / 3],
             [9 / 5, 7 / 5, 3 / 7, 1, 9 / 7], [8, 5 / 4, 3 / 4, 7 / 9, 1]]),
        'subcriteria': {
            'cost': {
                'matrix': np.array([[1, 1 / 7, 1 / 8], [7, 1, 1 / 3], [8, 3, 1]]),
                'subcriteria': {
                    'purchase_price': {
                        'matrix': np.array(
                            [[1, 5 / 7, 9 / 4, 5 / 4], [7 / 5, 1, 7 / 6, 6 / 7], [4 / 9, 6 / 7, 1, 2 / 3],
                             [4 / 5, 7 / 6, 3 / 2, 1]])
                    },
                    'fuel_cost': {
                        'matrix': np.array(
                            [[1, 3 / 7, 5 / 9, 1 / 2], [7 / 3, 1, 5 / 8, 5 / 8], [9 / 5, 8 / 5, 1, 1 / 2],
                             [2, 8 / 5, 2, 1]])
                    },
                    'maintenance_cost': {
                        'matrix': np.array([[1, 5 / 7, 3 / 4, 9 / 5], [7 / 5, 1, 1 / 2, 5 / 6], [4 / 3, 2, 1, 2 / 3],
                                            [5 / 9, 6 / 5, 3 / 2, 1]])
                    },
                }
            },
            'capacity': {
                'matrix': np.array([[1, 1 / 3], [3, 1]]),
                'subcriteria': {
                    'trunk_size': {
                        'matrix': np.array([[1, 5 / 6, 3 / 2, 2 / 5], [6 / 5, 1, 9 / 5, 5 / 7], [2 / 3, 5 / 9, 1, 1],
                                            [5 / 2, 7 / 5, 1, 1]])
                    },
                    'passenger_capacity': {
                        'matrix': np.array(
                            [[1, 1 / 9, 1 / 9, 8 / 3], [9, 1, 3 / 2, 9], [9, 2 / 3, 1, 9], [3 / 8, 1 / 9, 1 / 9, 1]])
                    },
                }
            },
            'design': {
                'matrix': np.array(
                    [[1, 9, 9, 9], [1 / 9, 1, 1 / 5, 8 / 9], [1 / 9, 5, 1, 9 / 7], [1 / 9, 9 / 8, 7 / 9, 1]]),
            },
            'safety': {
                'matrix': np.array(
                    [[1, 5 / 2, 9, 7], [2 / 5, 1, 9, 4], [1 / 9, 1 / 9, 1, 1 / 5], [1 / 7, 1 / 4, 5, 1]]),
            },
            'warranty': {
                'matrix': np.array(
                    [[1, 1 / 9, 3 / 4, 5 / 7], [9, 1, 9, 9], [4 / 3, 1 / 9, 1, 2], [7 / 5, 1 / 9, 1 / 2, 1]]),
            }
        }
    }
}


def calculate_weights(comparison_matrices, indent=0):
    for criteria in comparison_matrices:
        normalized_geometric = getWeightGM(comparison_matrices[criteria]['matrix'])
        max_eigenvalue, normalized_eigen = getWeightEigen(comparison_matrices[criteria]['matrix'])
        consistency_index, consistency_ratio = getCICR(max_eigenvalue, normalized_eigen)

        comparison_matrices[criteria]['normalized_geometric'] = normalized_geometric
        comparison_matrices[criteria]['normalized_eigen'] = normalized_eigen
        comparison_matrices[criteria]['consistency_index'] = consistency_index
        comparison_matrices[criteria]['consistency_ratio'] = consistency_ratio

        print(
            f'{"  " * indent}{criteria} CI: {np.real_if_close(consistency_index):.2f}, CR: {np.real_if_close(consistency_ratio):.2f}')

        if 'subcriteria' in comparison_matrices[criteria]:
            calculate_weights(comparison_matrices[criteria]['subcriteria'], indent + 1)


# returns ranking geometric, eigen
def calculate_ranking(comparison_matrix):
    if 'subcriteria' in comparison_matrix:
        geometric_rankings = []
        eigen_rankings = []
        for subcriteria in comparison_matrix['subcriteria']:
            geometric_ranking, eigen_ranking = calculate_ranking(comparison_matrix['subcriteria'][subcriteria])
            geometric_rankings.append(geometric_ranking)
            eigen_rankings.append(eigen_ranking)

        return np.matmul(np.stack(geometric_rankings, axis=-1), comparison_matrix['normalized_geometric']), np.matmul(
            np.stack(eigen_rankings, axis=-1), comparison_matrix['normalized_eigen'])
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
    geometric_ranking, eigenvalue_ranking = calculate_ranking(comparison_matrices['goal'])
    print(f'Geometric rankings: {np.real_if_close(geometric_ranking)}')
    print(f'Eigenvalue rankings: {np.real_if_close(eigenvalue_ranking)}')

    # (3) A typographical mistake in data entry results in the following criteria level comparison matrix
    # (the red fonts highlight the mistake). How does the ranking change? Is it possible to spot this
    # mistake BEFORE computing the ranking?
    print('\n######### 3) Typographic Mistake')
    comparison_matrices['goal']['matrix'] = np.array([[1, 5 / 7, 1 / 5, 5 / 9, 8], [7 / 5, 1, 5 / 9, 5 / 7, 4 / 5], [5, 9 / 5, 1, 7 / 3, 4 / 3], [9 / 5, 7 / 5, 3 / 7, 1, 9 / 7], [1 / 8, 5 / 4, 3 / 4, 7 / 9, 1]])
    calculate_weights(comparison_matrices)
    geometric_ranking, eigenvalue_ranking = calculate_ranking(comparison_matrices['goal'])
    print(f'Geometric rankings: {np.real_if_close(geometric_ranking)}')
    print(f'Eigenvalue rankings: {np.real_if_close(eigenvalue_ranking)}')