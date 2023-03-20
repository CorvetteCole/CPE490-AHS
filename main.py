import numpy
from comparison_matrix import ComparisonMatrix

if __name__ == "__main__":
    # create a new comparison matrix
    subcriterion = [ComparisonMatrix('Size', numpy.array([[1, 5], [1 / 5, 1]])),
                    ComparisonMatrix('Thermal Management', numpy.array([[1, 6], [1 / 6, 1]])),
                    ComparisonMatrix('Accessibility', numpy.array([[1, 6 / 5], [5 / 6, 1]])),
                    ComparisonMatrix('Cost', numpy.array([[1, 2], [1 / 2, 1]]))]

    pcbake = ComparisonMatrix('PCBake', numpy.array(
        [[1, 1 / 4, 1 / 3, 1 / 2], [4, 1, 4 / 3, 2], [3, 3 / 4, 1, 6 / 5], [2, 1 / 2, 5 / 6, 1]]), subcriterion)

    print(pcbake)

    print(pcbake.get_consistency_str())

    print(pcbake.get_ranking_str())
