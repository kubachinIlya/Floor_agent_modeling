import matplotlib.pyplot as plt
from random import randint
import os
import numpy as np
# переменная для задания кол-ва агентов на графике
from Agents import number_of_agents


# функция берет из файла координаты агента и сохраняет их в два массива
def reading_coordinates(filename):
    x_coordinates_f = []
    y_coordinates_f = []
    text_file = open(r".\way_points_history_for_agents\agent" + str(filename) + ".txt", "r")
    lines = text_file.read().split(' ')
#    print(lines)
    for i in range(0, len(lines) - 1, 2):
        x_coordinates_f.append(float(lines[i]))
        y_coordinates_f.append(float(lines[i + 1]))
    text_file.close()
    return x_coordinates_f, y_coordinates_f


def reading_files(filename):
    x_coordinates_f = []
    y_coordinates_f = []
    text_file = open(filename)
    lines = text_file.read().split(' ')
#    print(lines)
    for i in range(0, len(lines) - 1, 2):
        x_coordinates_f.append(float(lines[i]))
        y_coordinates_f.append(float(lines[i + 1]))
    text_file.close()
    return x_coordinates_f, y_coordinates_f


# задание кол-ва агента на графике
print("input number of agents for a plot, no more than ", number_of_agents)
number_of_agents_for_a_plot = int(input())
if number_of_agents_for_a_plot > number_of_agents:
    number_of_agents_for_a_plot = number_of_agents

# добавление координат сенсоров для использования на графике
array = np.load(
    "Src\Src\medicine-data\iBeacon_data\summer_2_floor\points_wifi_2.npy")
sensors_x = []
sensors_y = []
for i in range(0, 9):
    x_coord = array[i][0]
    y_coord = array[i][1]
    sensors_x.append(x_coord - 235)
    sensors_y.append(y_coord - 76)

# цвета для всех агентов на графике
colors = []
for i in range(number_of_agents):
    colors.append('#%06X' % randint(0, 0xFFFFFF))

# Загрузка фона для графика ( картинка больницы)
img = plt.imread('map_2floor_bw.png')
fig, ax = plt.subplots()
ax.imshow(img)

# помещаем на график начальных агентов
for i in range(0, number_of_agents_for_a_plot):
    x_coordinates, y_coordinates = reading_coordinates(i)
    plt.scatter(x_coordinates, y_coordinates, color=str(colors[i]), s=10)
    plt.scatter(sensors_x, sensors_y, color='r', s=30)
    # plt.plot(x_coordinates, y_coordinates, color=str(colors[i]))

# узнаем сколько всего агентов наплодилось
directory = "way_points_history_for_agents"  # Укажите путь к нужной директории

# Получаем список всех файлов и директорий в указанной директории
files = os.listdir(directory)

# Подсчитываем количество файлов
file_count = len(files)

# помещаем на график агентов - клонов
for i in range(number_of_agents_for_a_plot, file_count-1):
    x_coordinates, y_coordinates = reading_coordinates(i)
    plt.scatter(x_coordinates, y_coordinates, color='black', s=10)
    # plt.scatter(sensors_x, sensors_y, color='r', s=30)
    # plt.plot(x_coordinates, y_coordinates, color=str(colors[i]))


x_coordinates_ideal, y_coordinates_ideal = reading_coordinates(99999)
print(x_coordinates_ideal)
# plt.scatter(x_coordinates_ideal, y_coordinates_ideal, color='blue', marker='s', s=20)
plt.scatter(x_coordinates_ideal, y_coordinates_ideal, color='blue', marker='s', s=20, alpha=0.5)
plt.plot(x_coordinates_ideal, y_coordinates_ideal, color='blue', alpha=0.5)

x_triangulation, y_triangulation = reading_files("triangulation.txt")
plt.scatter(x_triangulation, y_triangulation, color='pink', marker='s', s=30, alpha=0.5)
plt.plot(x_triangulation, y_triangulation, color='pink', alpha=0.5)

# Задаем оси для графика
plt.ylabel('y_coordinate_of_agent')
plt.xlabel('x_coordinate_of_agent')
plt.show()

# удаление файлов истории агентов
print("Do you want to remove agents_history files? yes/no")
answer = input()
if answer == 'yes':
    for i in range(0, file_count-1):
        os.remove(r".\way_points_history_for_agents\/agent" + str(i) + ".txt")
    os.remove(r".\way_points_history_for_agents/agent99999.txt")
    os.remove(r".\counter_new_waves.txt")
    print("files have been removed")
else:
    print("files still there")