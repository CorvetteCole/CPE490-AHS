import numpy
from typing import List

random_index = [0, 0, 0.52, 0.89, 1.11, 1.25, 1.35, 1.4, 1.45, 1.49, 1.52]


class ComparisonMatrix:
    # list of self
    subcriteria: List['ComparisonMatrix']

    def __init__(self, matrix, subcriteria=None):
        self.matrix = matrix
        if subcriteria is None:
            self._subcriteria = []
        else:
            self.subcriteria = subcriteria

    @property
    def subcriteria(self):
        return self._subcriteria

    @subcriteria.setter
    def subcriteria(self, value):
        # validate, then recalculate all consistency, ranking, weights
        pass

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        # validate, then recalculate all consistency, ranking, weights
        self._matrix = value
        pass




    @property
    def has_subcriteria(self):
        return len(self.subcriteria) > 0

    @property
    def ranking_geometric(self):
        if self.has_subcriteria:
            return numpy.matmul(numpy.stack([s.ranking_geometric for s in self.subcriteria], axis=-1),
                                self.normalized_geometric)
        else:
            return self.normalized_geometric

    @property
    def ranking_eigen(self):
        if self.has_subcriteria:
            return numpy.matmul(numpy.stack([s.ranking_eigen for s in self.subcriteria], axis=-1),
                                self.normalized_eigen)
        else:
            return self.normalized_eigen

    
    @property
    def normalized_geometric(self) -> numpy.ndarray:
        """
        Compute normalized weights based on Geometric Mean method

        :return: the normalized weight vector
        """
        column_sum = numpy.tile(numpy.sum(self.matrix, 0), [self.matrix.shape[0], 1])
        normalized_matrix = numpy.divide(self.matrix, column_sum)
        geometric_weight = numpy.prod(normalized_matrix, 1) ** (1.0 / self.matrix.shape[1])
        return geometric_weight / numpy.sum(geometric_weight)

    @property
    def normalized_eigen(self) -> numpy.ndarray:
        """
        Compute normalized weights based on Eigen method

        :return: the normalized weight vector
        """
        eigenvalues, normalized_eigenvectors = numpy.linalg.eig(self.matrix)
        # get max eigenvalue and corresponding eigenvector
        max_eigenvalue_index = numpy.argmax(eigenvalues)
        # where max_eigenvector is the eigenvector corresponding to the maximum eigenvalue
        max_eigenvector = normalized_eigenvectors[:, max_eigenvalue_index]
        return max_eigenvector / numpy.sum(max_eigenvector)

    @property
    def _max_eigenvalue(self) -> float:
        return numpy.max(numpy.linalg.eig(self.matrix)[0])

    @property
    def consistency_index(self):
        return (self._max_eigenvalue - self.matrix.shape[0]) / (self.matrix.shape[0] - 1)

    @property
    def consistency_ratio(self):
        return None
