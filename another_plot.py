import matplotlib.pyplot as plt
from random import randint
import os
import numpy as np
import math

# нужно прочитать координаты по триангуляции
# нужно прочитать координаты по линейной регресси
# найти между ними расстояние
# построить график расстояние - шаг идеального агента

def reading_coordinates(filename):
    x_coordinates_f = []
    y_coordinates_f = []
    text_file = open(filename, "r")
    lines = text_file.read().split(' ')
#    print(lines)
    for i in range(0, len(lines) - 1, 2):
        x_coordinates_f.append(float(lines[i]))
        y_coordinates_f.append(float(lines[i + 1]))
    text_file.close()
    return x_coordinates_f, y_coordinates_f

def reading_ideal(filename):
    step_number = []
    text_file = open(filename, "r")
    lines = text_file.read().split(' ')
#    print(lines)
    for i in range(0, len(lines) - 1, 1):
        step_number.append(float(lines[i]))
    text_file.close()
    return step_number


distances = []

triang_x, triang_y = reading_coordinates("triangulation.txt")
mean_x, mean_y = reading_coordinates("mean_position.txt")
ideal_step = reading_ideal("ideal_agent.txt")

for i in range(0, len(triang_x)):
    d = math.sqrt((mean_x[i] - triang_x[i])**2 + (mean_y[i] - triang_y[i])**2)
    distances.append(d)

ideal_step = ideal_step[:(len(distances))]

plt.scatter(ideal_step, distances, color='blue', marker='s', s=20, alpha=0.5)
plt.plot(ideal_step, distances, color='blue', alpha=0.5)
plt.ylabel('Distance')
plt.xlabel('Time')
plt.show()
