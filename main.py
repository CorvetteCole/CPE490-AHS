import numpy
from comparison_matrix import ComparisonMatrix

if __name__ == "__main__":
    # legacy data structure:
    # comparison_matrices = {
    #     'goal': {
    #         'matrix': np.array(
    #             [[1, 5 / 7, 1 / 5, 5 / 9, 1 / 8], [7 / 5, 1, 5 / 9, 5 / 7, 4 / 5], [5, 9 / 5, 1, 7 / 3, 4 / 3],
    #              [9 / 5, 7 / 5, 3 / 7, 1, 9 / 7], [8, 5 / 4, 3 / 4, 7 / 9, 1]]),
    #         'subcriteria': {
    #             'cost': {
    #                 'matrix': np.array([[1, 1 / 7, 1 / 8], [7, 1, 1 / 3], [8, 3, 1]]),
    #                 'subcriteria': {
    #                     'purchase_price': {
    #                         'matrix': np.array(
    #                             [[1, 5 / 7, 9 / 4, 5 / 4], [7 / 5, 1, 7 / 6, 6 / 7], [4 / 9, 6 / 7, 1, 2 / 3],
    #                              [4 / 5, 7 / 6, 3 / 2, 1]])
    #                     },
    #                     'fuel_cost': {
    #                         'matrix': np.array(
    #                             [[1, 3 / 7, 5 / 9, 1 / 2], [7 / 3, 1, 5 / 8, 5 / 8], [9 / 5, 8 / 5, 1, 1 / 2],
    #                              [2, 8 / 5, 2, 1]])
    #                     },
    #                     'maintenance_cost': {
    #                         'matrix': np.array([[1, 5 / 7, 3 / 4, 9 / 5], [7 / 5, 1, 1 / 2, 5 / 6], [4 / 3, 2, 1, 2 / 3],
    #                                             [5 / 9, 6 / 5, 3 / 2, 1]])
    #                     },
    #                 }
    #             },
    #             'capacity': {
    #                 'matrix': np.array([[1, 1 / 3], [3, 1]]),
    #                 'subcriteria': {
    #                     'trunk_size': {
    #                         'matrix': np.array([[1, 5 / 6, 3 / 2, 2 / 5], [6 / 5, 1, 9 / 5, 5 / 7], [2 / 3, 5 / 9, 1, 1],
    #                                             [5 / 2, 7 / 5, 1, 1]])
    #                     },
    #                     'passenger_capacity': {
    #                         'matrix': np.array(
    #                             [[1, 1 / 9, 1 / 9, 8 / 3], [9, 1, 3 / 2, 9], [9, 2 / 3, 1, 9], [3 / 8, 1 / 9, 1 / 9, 1]])
    #                     },
    #                 }
    #             },
    #             'design': {
    #                 'matrix': np.array(
    #                     [[1, 9, 9, 9], [1 / 9, 1, 1 / 5, 8 / 9], [1 / 9, 5, 1, 9 / 7], [1 / 9, 9 / 8, 7 / 9, 1]]),
    #             },
    #             'safety': {
    #                 'matrix': np.array(
    #                     [[1, 5 / 2, 9, 7], [2 / 5, 1, 9, 4], [1 / 9, 1 / 9, 1, 1 / 5], [1 / 7, 1 / 4, 5, 1]]),
    #             },
    #             'warranty': {
    #                 'matrix': np.array(
    #                     [[1, 1 / 9, 3 / 4, 5 / 7], [9, 1, 9, 9], [4 / 3, 1 / 9, 1, 2], [7 / 5, 1 / 9, 1 / 2, 1]]),
    #             }
    #         }
    #     }
    # }

    purchase_price = ComparisonMatrix('Purchase Price', numpy.matrix(
        [[1, 7 / 5, 4 / 9, 4 / 5],
         [5 / 7, 1, 6 / 7, 7 / 6],
         [9 / 4, 7 / 6, 1, 3 / 2],
         [5 / 4, 6 / 7, 2 / 3, 1]]))

    fuel_cost = ComparisonMatrix('Fuel Cost', numpy.matrix(
        [[1, 7 / 3, 9 / 5, 2],
         [3 / 7, 1, 8 / 5, 8 / 5],
         [5 / 9, 5 / 8, 1, 2],
         [1 / 2, 5 / 8, 1 / 2, 1]]))

    maintenance_cost = ComparisonMatrix('Maintenance Cost', numpy.matrix(
        [[1, 7 / 5, 4 / 3, 5 / 9],
         [5 / 7, 1, 2, 6 / 5],
         [3 / 4, 1 / 2, 1, 3 / 2],
         [9 / 5, 5 / 6, 2 / 3, 1]]))

    trunk_size = ComparisonMatrix('Trunk Size', numpy.matrix(
        [[1, 6 / 5, 2 / 3, 5 / 2],
         [5 / 6, 1, 5 / 9, 7 / 5],
         [3 / 2, 9 / 5, 1, 1],
         [2 / 5, 5 / 7, 1, 1]]))

    passenger_capacity = ComparisonMatrix('Passenger Capacity', numpy.matrix(
        [[1, 9, 9, 3 / 8],
         [1 / 9, 1, 2 / 3, 1 / 9],
         [1 / 9, 3 / 2, 1, 1 / 9],
         [8 / 3, 9, 9, 1]]))

    safety = ComparisonMatrix('Safety', numpy.matrix(
        [[1, 2 / 5, 1 / 9, 1 / 7],
         [5 / 2, 1, 1 / 9, 1 / 4],
         [9, 9, 1, 5],
         [7, 4, 1 / 5, 1]]))

    design = ComparisonMatrix('Design', numpy.matrix(
        [[1, 1 / 9, 1 / 9, 1 / 9],
         [9, 1, 5, 9 / 8],
         [9, 1 / 5, 1, 7 / 9],
         [9, 8 / 9, 9 / 7, 1]]))

    warranty = ComparisonMatrix('Warranty', numpy.matrix(
        [[1, 9, 4 / 3, 7 / 5],
         [1 / 9, 1, 1 / 9, 1 / 9],
         [3 / 4, 9, 1, 1 / 2],
         [5 / 7, 9, 2, 1]]))

    cost = ComparisonMatrix('Cost', numpy.matrix(
        [[1, 7, 8],
         [1 / 7, 1, 3],
         [1 / 8, 1 / 3, 1]]), subcriteria=[purchase_price, fuel_cost, maintenance_cost])

    capacity = ComparisonMatrix('Capacity', numpy.matrix(
        [[1, 3],
         [1 / 3, 1]]), subcriteria=[passenger_capacity, trunk_size])

    goal = ComparisonMatrix('Car', numpy.matrix(
        [[1, 7 / 5, 5, 9 / 5, 8],
         [5 / 7, 1, 9 / 5, 7 / 5, 5 / 4],
         [1 / 5, 5 / 9, 1, 3 / 7, 3 / 4],
         [5 / 9, 5 / 7, 7 / 3, 1, 7 / 9],
         [1 / 8, 4 / 5, 4 / 3, 9 / 7, 1]]), subcriteria=[cost, safety, design, capacity, warranty])

    print(goal)

    print(goal.get_consistency_str())

    print(goal.get_ranking_str())
