import numpy
from comparison_matrix import ComparisonMatrix

if __name__ == "__main__":
    # cost 	Door 	display	E-stop	Disassembly
    # Door 	1	3	1/2.	5
    # E.stop	1/3.	1	3	3
    # display 	2	1/3.	1	4
    # Dissasembly 	1/5.	1/3.	1/4.	1

    cost = ComparisonMatrix('cost', numpy.matrix(
        [[1, 3, 0.5, 5],
         [1 / 3, 1, 3, 3],
         [2, 1 / 3, 1, 4],
         [1 / 5, 1 / 3, 1 / 4, 1]]))

    # Functionality	Door 	E-stop	display 	Disassembly
    # Door 	1	6	8	1
    # display 	1/6.	1	2	1/5.
    # E-stop 	1/8.	1/2.	1	1/4.
    # Dissasembly 	1	5	4	1

    functionality = ComparisonMatrix('functionality', numpy.matrix(
        [[1, 6, 8, 1],
         [1 / 6, 1, 2, 1 / 5],
         [1 / 8, 1 / 2, 1, 1 / 4],
         [1, 5, 4, 1]]))

    # Exstadibility	Door	display	E-stop	Disasembly Geometric
    # Door	1	2	4	5
    # Display	1/2.	1	2	3
    # E-stop	1/4.	1/2.	1	3
    # Disassembly	1/5.	1/3.	1/3.	1

    extensibility = ComparisonMatrix('extensibility', numpy.matrix(
        [[1, 2, 4, 5],
         [1 / 2, 1, 2, 3],
         [1 / 4, 1 / 2, 1, 3],
         [1 / 5, 1 / 3, 1 / 3, 1]]))

    # critiria 	Cost 	Functionality	exstandibility
    # cost 	1	1/5.	1/3.
    # functionality 	5	3	2
    # exstandibility 	3	1/2.	1

    criteria = ComparisonMatrix('criteria', numpy.matrix(
        [[1, 1 / 5, 1 / 3],
         [5, 3, 2],
         [3, 1 / 2, 1]]), [cost, functionality, extensibility])

    print(criteria.get_consistency_str())
    print(criteria.get_ranking_str())
