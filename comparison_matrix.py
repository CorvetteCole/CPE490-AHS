import numpy
import copy
from typing import List

random_index = [0, 0, 0.52, 0.89, 1.11, 1.25, 1.35, 1.4, 1.45, 1.49, 1.52]


def normalize_geometric(matrix) -> numpy.ndarray:
    """
    Compute normalized weights based on Geometric Mean method

    :return: the normalized weight vector
    """
    column_sum = numpy.tile(numpy.sum(matrix, 0), [matrix.shape[0], 1])
    normalized_matrix = numpy.divide(matrix, column_sum)
    geometric_weight = numpy.prod(normalized_matrix, 1) ** (1.0 / matrix.shape[1])
    return geometric_weight / numpy.sum(geometric_weight)


def normalize_eigen(matrix) -> numpy.ndarray:
    """
    Compute normalized weights based on Eigen method

    :return: the normalized weight vector
    """
    eigenvalues, normalized_eigenvectors = numpy.linalg.eig(matrix)
    # get max eigenvalue and corresponding eigenvector
    max_eigenvalue_index = numpy.argmax(eigenvalues)
    # where max_eigenvector is the eigenvector corresponding to the maximum eigenvalue
    max_eigenvector = normalized_eigenvectors[:, max_eigenvalue_index]
    return max_eigenvector / numpy.sum(max_eigenvector)


def calculate_consistency(matrix, normalized_eigen):
    max_eigenvalue = numpy.max(numpy.linalg.eig(matrix)[0])
    consistency_index = (max_eigenvalue - matrix.shape[0]) / (matrix.shape[0] - 1)
    consistency_ratio = consistency_index / random_index[normalized_eigen.shape[0]]
    return consistency_index, consistency_ratio


class ComparisonMatrix:
    # list of self
    _subcriteria: List['ComparisonMatrix'] = []
    _matrix: numpy.ndarray

    _normalized_geometric: numpy.ndarray
    _normalized_eigen: numpy.ndarray
    _consistency_index: float
    _consistency_ratio: float

    def __init__(self, name: str, matrix: numpy.ndarray, subcriteria: List['ComparisonMatrix'] = None):
        self.name = name
        self.matrix = matrix
        self.subcriteria = subcriteria

    @property
    def subcriteria(self) -> List['ComparisonMatrix']:
        """ :return: a copy of the subcriteria list, so that it can't be modified """
        return copy.deepcopy(self._subcriteria)

    @subcriteria.setter
    def subcriteria(self, value: List['ComparisonMatrix']):
        """
        :param value: a list of ComparisonMatrix
        :raises: ValueError if value is not a list of ComparisonMatrix
        """
        # validate, ensure that this is a list of ComparisonMatrix

        if value is None:
            self._subcriteria = []
            return
        elif isinstance(value, list):
            for v in value:
                if not isinstance(v, ComparisonMatrix):
                    raise ValueError('subcriteria must be a list of ComparisonMatrix')
        else:
            raise ValueError('subcriteria must be a list of ComparisonMatrix')

        self._subcriteria = value
        self._recalculate_weights()

    @property
    def matrix(self) -> numpy.ndarray:
        """ :return: a copy of the matrix, so that it can't be modified """
        return copy.deepcopy(self._matrix)

    @matrix.setter
    def matrix(self, value: numpy.ndarray):
        """
        :param value: a numpy.ndarray
        :raises: ValueError if value is not a numpy.ndarray
        """

        if not isinstance(value, numpy.ndarray):
            raise ValueError('matrix must be a numpy.ndarray')

        self._matrix = value
        self._recalculate_weights()

    @property
    def has_subcriteria(self):
        return len(self.subcriteria) > 0

    @property
    def ranking_geometric(self):
        if self.has_subcriteria:
            return numpy.matmul(numpy.stack([s.ranking_geometric for s in self.subcriteria], axis=-1),
                                self._normalized_geometric)
        else:
            return self._normalized_geometric

    @property
    def ranking_eigen(self):
        if self.has_subcriteria:
            return numpy.matmul(numpy.stack([s.ranking_eigen for s in self.subcriteria], axis=-1),
                                self._normalized_eigen)
        else:
            return self._normalized_eigen

    def _recalculate_weights(self):
        self._normalized_geometric = normalize_geometric(self.matrix)
        self._normalized_eigen = normalize_eigen(self.matrix)
        self._consistency_index, self._consistency_ratio = calculate_consistency(self.matrix, self._normalized_eigen)

    @property
    def consistency_index(self):
        return self._consistency_index

    @property
    def consistency_ratio(self):
        return self._consistency_ratio

    def __str__(self):
        return str(self.matrix)

    def get_consistency_str(self):
        string_representation = f'{self.name} | ' \
                                f'CI: {numpy.real_if_close(self.consistency_index):.2f} ' \
                                f'CR: {numpy.real_if_close(self.consistency_ratio):.2f}\n'
        if self.has_subcriteria:
            for subcriteria in self.subcriteria:
                string_representation += f'  {subcriteria.get_consistency_str()}'

        return string_representation

    def get_ranking_str(self):
        # round to 2 floating points for each value in numpy arrays
        geometric_rounded = numpy.round(numpy.real_if_close(self.ranking_geometric), 2)
        eigen_rounded = numpy.round(numpy.real_if_close(self.ranking_eigen), 2)
        string_representation = f'{self.name} | Geometric: {geometric_rounded} Eigen: {eigen_rounded}\n'
        if self.has_subcriteria:
            for subcriteria in self.subcriteria:
                string_representation += f'  {subcriteria.get_ranking_str()}'

        return string_representation
