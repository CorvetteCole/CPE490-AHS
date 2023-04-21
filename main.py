import numpy
from comparison_matrix import ComparisonMatrix

if __name__ == "__main__":
    # Cost
    # Single
    # Separate
    # Single
    # 1
    # 2
    # Separate
    # 1/2
    # 1
    cost = ComparisonMatrix('Cost', numpy.matrix(
        [[1, 3 / 4],
         [4 / 3, 1]]))

    print(cost.get_consistency_str())
    print(cost.get_ranking_str())

    # Safety
    # Single
    # Separate
    # Single
    # 1
    # 1/3
    # Separate
    # 3
    # 1

    safety = ComparisonMatrix('Safety', numpy.matrix(
        [[1, 1 / 3],
         [3, 1]]))

    print(safety.get_consistency_str())
    print(safety.get_ranking_str())

    # Performance
    # Single
    # Separate
    # Single
    # 1
    # 3/4
    # Separate
    # 4/3
    # 1

    performance = ComparisonMatrix('Performance', numpy.matrix(
        [[1, 3 / 4],
         [4 / 3, 1]]))

    print(performance.get_consistency_str())
    print(performance.get_ranking_str())

    # Complexity
    # Single
    # Separate
    # Single
    # 1
    # 4/3
    # Separate
    # 3/4
    # 1

    complexity = ComparisonMatrix('Complexity', numpy.matrix(
        [[1, 4 / 3],
         [3 / 4, 1]]))

    print(complexity.get_consistency_str())
    print(complexity.get_ranking_str())

    # Criteria
    # Cost
    # Safety
    # Performance
    # Complexity
    # Cost
    # 1
    # 3
    # 2
    # 4
    # Safety
    # 1/3
    # 1
    # 1/2
    # 1/2
    # Performance
    # 1/2
    # 2
    # 1
    # 3
    # Complexity
    # 1/4
    # 2
    # 1/3
    # 1

    criteria = ComparisonMatrix('Criteria', numpy.matrix(
        [[1, 3, 2, 4],
         [1 / 3, 1, 1 / 2, 1 / 2],
         [1 / 2, 2, 1, 3],
         [1 / 4, 2, 1 / 3, 1]]), [cost, safety, performance, complexity])

    print(criteria.get_consistency_str())
    print(criteria.get_ranking_str())
