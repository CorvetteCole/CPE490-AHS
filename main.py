import numpy
from comparison_matrix import ComparisonMatrix

if __name__ == "__main__":
    # Time Consumption	Open	Closed
    # Open 	1	2
    # Closed	(1/2)	1

    time_consumption = ComparisonMatrix("Time Consumption", numpy.matrix(
        [[1, 2],
         [0.5, 1]]))

    # Accuracy	Open	Closed
    # Open 	1	3
    # Closed	(1/3)	1

    accuracy = ComparisonMatrix("Accuracy", numpy.matrix(
        [[1, 3],
         [1 / 3, 1]]))

    # 	Time Consumption	Accuracy
    # Time Consumption	1	(2/3)
    # Accuracy	(3/2)	1

    control_loop_type = ComparisonMatrix("Control Loop Type", numpy.matrix(
        [[1, 2 / 3],
         [3 / 2, 1]]), [time_consumption, accuracy])

    print(control_loop_type.get_consistency_str())

    print(control_loop_type.get_ranking_str())
