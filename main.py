import numpy
from comparison_matrix import ComparisonMatrix

if __name__ == "__main__":
    # 	Weights
    # Cost	0.35
    # Robustness	0.65

    # Cost	Hall Effect	Contact	Infrared
    # Hall Effect	1.00	0.75	0.40
    # Contact	1.33	1.00	1.50
    # Infrared	2.50	0.66	1.00

    # Robustness	Hall Effect	Contact	Infrared
    # Hall Effect	1.00	0.60	0.95
    # Contact	1.66	1.00	1.50
    # Infrared	1.05	0.66	1.00

    door_sensor_cost = ComparisonMatrix("Door Sensor Cost", numpy.matrix(
        [[1, 0.75, 0.40],
         [1.33, 1, 1.5],
         [2.5, 0.66, 1]]))
    door_sensor_robustness = ComparisonMatrix("Door Sensor Robustness", numpy.matrix(
        [[1, 0.6, 0.95],
         [1.66, 1, 1.5],
         [1.05, 0.66, 1]]))

    door_sensor = ComparisonMatrix("Door Sensor", numpy.matrix(
        [[1, 1.85],
         [0.53, 1]]), subcriteria=[door_sensor_cost, door_sensor_robustness])

    print(door_sensor.get_consistency_str())
    print(door_sensor.get_ranking_str())

    # Temperature Sensor Tables
    #
    # Cost	RTD	NTC Glass	NTC Bead
    # RTD	1.00	2.50	3.00
    # NTC Glass	0.40	1.00	1.20
    # NTC Bead	0.33	0.83	1.00
    #
    # Robustness	RTD	NTC Glass	NTC Bead
    # RTD	1.00	0.20	0.30
    # NTC Glass	5.00	1.00	1.20
    # NTC Bead	3.33	0.83	1.00

    temperature_sensor_cost = ComparisonMatrix("Temperature Sensor Cost", numpy.matrix(
        [[1, 2.5, 3],
         [0.4, 1, 1.2],
         [0.33, 0.83, 1]]))

    temperature_sensor_robustness = ComparisonMatrix("Temperature Sensor Robustness", numpy.matrix(
        [[1, 0.2, 0.3],
         [5, 1, 1.2],
         [3.33, 0.83, 1]]))

    temperature_sensor = ComparisonMatrix("Temperature Sensor", numpy.matrix(
        [[1, 1.85],
         [0.53, 1]]), subcriteria=[temperature_sensor_cost, temperature_sensor_robustness])

    print(temperature_sensor.get_consistency_str())
    print(temperature_sensor.get_ranking_str())
