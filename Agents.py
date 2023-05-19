import math
import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np
import json
import re
import random
import math
from random import randint
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector
from math import sqrt, fabs
from scipy.spatial import Delaunay

# Задаем количество агентов, соответственно количество частиц (это число нужно задать и в самой модели)
number_of_agents = num_particles = 3


# класс окружения агента (физической среды)
class AgentEnvironmentMap:
    def __init__(self, bw_image_path, dist_per_pix=1.0):
        self.img = image.imread(bw_image_path)
        self.shape = (self.img.shape[0], self.img.shape[1])
        self.dist_per_pix = dist_per_pix
        self.max_x = self.shape[1] * dist_per_pix - 1
        self.max_y = self.shape[0] * dist_per_pix - 1
        self.wall_mask = np.zeros(self.shape, dtype=bool)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.wall_mask[i, j] = self.img[i, j, 0] < 0.1

    def is_wall(self, x, y):
        i = int(y / self.dist_per_pix)
        j = int(x / self.dist_per_pix)
        return self.wall_mask[i, j]

    def to_ascii1(self):
        return '\n'.join([''.join(['#' if m else '.' for m in ln]) for ln in self.wall_mask])


# класс агента
class PhysicalAgent(Agent):
    def __init__(self, unique_id, model, speed_of_agent):
        self.is_moving = True
        super().__init__(unique_id, model)
        self.speed_of_agent = speed_of_agent

    def reset_waypoints(self, waypoints=None):
        if waypoints:
            self.waypoints = waypoints
        self.model.space.move_agent(self, self.waypoints[0])
        self.next_waypoint_index = 1
        self.is_moving = True

    def get_points_to_show(self):
        return {'agent': self.pos, 'final_target': self.waypoints[-1],
                'next_target': self.waypoints[self.next_waypoint_index]}

    # возвращает скорость агента c рандомным отклонением
    def speed_agent(self):
        random_speed_change = random.uniform(-0.25, 0.25)  # рандомное отклонение агента(ГРАФИК)
        return self.speed_of_agent + random_speed_change

    def step(self):

        TARGET_SENSITIVITY = 7
        search_target = True
        while search_target:
            dx = self.waypoints[self.next_waypoint_index][0] - self.pos[0]
            dy = self.waypoints[self.next_waypoint_index][1] - self.pos[1]
            dx = ((dx * (math.cos(math.radians(randint(-5, 5))))) + (dy * (math.sin(math.radians(randint(-5, 5))))))  # Отклонение вектора
            dy = ((dx * (-math.sin(math.radians(randint(-5, 5))))) + (dy * (math.cos(math.radians(randint(-5, 5))))))  # ГРАФИК
            d = np.sqrt(dx * dx + dy * dy)
            if d < TARGET_SENSITIVITY:
                if self.next_waypoint_index < len(self.waypoints) - 1:
                    self.next_waypoint_index += 1
                else:
                    self.is_moving = False
                    return
            else:
                search_target = False
        new_x = self.pos[0] + self.speed_agent() * dx / d
        new_y = self.pos[1] + self.speed_agent() * dy / d
        if not self.model.env_map.is_wall(new_x, new_y):
            print(
                f'#{self.unique_id} is moving to ({new_x}, {new_y}) forwards waypoint #{self.next_waypoint_index} with the speed {self.speed_agent()}')
            # print(self.pos)
            self.model.space.move_agent(self, (new_x, new_y))

        # запись истории путешествий агента
        my_file = open(r".\way_points_history_for_agents\agent" + str(self.unique_id) + ".txt", "a")
        my_file.write(str(new_x) + " " + str(new_y) + " ")
        my_file.close()


class MySensor:
    def __init__(self, unique_id, pos):
        self.unique_id = unique_id
        self.speed_sensor_radius = 600
        self.pos = pos
        self.array_of_signals = [[] * 1 for _ in range(number_of_agents)]  # массив для хранения сигналов агентов

    def sense_signal(self, agents):
        step_to_go_x = 0.3  # шаг для прохода расстояния между положениями агента по X
        # создаем экземпляр пространства для функции
        map_for_wall = AgentEnvironmentMap('map_2floor_bw.png')
        for agent in agents:
            if agent.unique_id >= len(self.array_of_signals):
                self.array_of_signals.append([])
            # начальная и конечная позиции агентов на шаге
            x_starting_point = self.pos[0]
            y_starting_point = self.pos[1]
            x_end_point = agent.pos[0]
            y_end_point = agent.pos[1]

            steps_between_pos = round(fabs((x_end_point - x_starting_point) / step_to_go_x))

            # считаем какой нужен шаг по y
            if steps_between_pos != 0:
                step_to_go_y = (y_end_point - y_starting_point) / steps_between_pos
            else:
                step_to_go_y = 0.3  # в такой ситуации по идее шагов не будет и игрек не изменится

            # смотрит какой по знаку шаг по x нам нужен
            if x_end_point < x_starting_point:
                step_to_go_x = - 0.3
            else:
                step_to_go_x = 0.3

            level_of_signal_for_array = 100

            for i in range(0, steps_between_pos):
                x_starting_point += step_to_go_x
                y_starting_point += step_to_go_y
                if AgentEnvironmentMap.is_wall(map_for_wall, x_starting_point, y_starting_point):
                    level_of_signal_for_array -= 0.3  # ЗА СТЕНУ ОТНЯЛИ 0.3 от сигнала
                else:
                    level_of_signal_for_array -= 0.1  # ЗА ПРОСТРАНСТВО 0.1 от сигнала

            self.array_of_signals[agent.unique_id].append(level_of_signal_for_array)
        print(*self.array_of_signals)
        print()





class IndoorModel(Model):
    def __init__(self, agents_json_path='agents.json', env_map_path='map_2floor_bw.png'):
        super().__init__()
        hard_dx = 235
        hard_dy = 76
        self.path = env_map_path
        self.env_map = AgentEnvironmentMap(env_map_path)
        self.space = ContinuousSpace(self.env_map.max_x, self.env_map.max_y, False)
        self.schedule = RandomActivation(self)
        self.sensors_arr = []
        self.number_ag = 3

        with open(agents_json_path) as f:
            agent_json = json.load(f)

        for k in range(0, number_of_agents):
            my_file = open(r".\way_points_history_for_agents\agent" + str(k) + ".txt", "w+")
            my_file.close()
            for i, aj in enumerate(agent_json):
                with open(aj['waypoints_path']) as f:
                    lns = f.readlines()
                    waypoints = []
                    for ln in lns:
                        parts = re.findall('\d+', ln)
                        waypoints.append((int(parts[0].strip()) - hard_dx, int(parts[1].strip()) - hard_dy))
                    speed = random.uniform(1, 6)
                    a = PhysicalAgent(k, self, speed)
                    self.schedule.add(a)
                    self.space.place_agent(a, waypoints[0])
                    a.reset_waypoints(waypoints)

        # добавлены сенсоры из файла
        array = np.load(
            "Src\Src\medicine-data\iBeacon_data\summer_2_floor\points_wifi_2.npy")

        for i in range(0, 9):
            x_coord = array[i][0]
            y_coord = array[i][1]
            data_datchik = MySensor(i, (x_coord - hard_dx, y_coord - hard_dy))
            self.sensors_arr.append(data_datchik)

        self.data_collector = DataCollector({'moving_agents_num': 'moving_agents_num'},
                                            {'is_moving': 'is_moving', 'x': lambda a: a.pos[0],
                                             'y': lambda a: a.pos[1]})

        self.moving_agents_num = 0
        self.running = True
        self.data_collector.collect(self)
        self.agents = []
        # self.agents = self.space.get_neighbors((250, 125), 500, True)  # сбор агентов для анализа датчиками

        self.step_number = 0  # шаг модели для вызова фильтра частиц на определенных шагах

    def step(self):
        self.schedule.step()
        self.data_collector.collect(self)
        self.step_number += 1
        # print(self.step_number) вывод номера шага модели
        self.moving_agents_num = sum([a.is_moving for a in self.schedule.agents])
        self.running = self.moving_agents_num > 0

        # вызов фильтра частиц на каждом определенном шаге
        if self.step_number % 5 == 0:
            self.particle_filter()
            print('СРАБОТАЛ ФИЛЬТР')

        # вызов сенсоров для работы по определению уровня сигнала
        self.agents = self.space.get_neighbors((250, 125), 500, True)  # сбор агентов для анализа датчиками
        for sens in self.sensors_arr:
            sens.sense_signal(self.agents)

    def plot_explicitly(self):
        plt.imshow(self.env_map.img)
        for a in self.schedule.agents:
            plt.plot(a.pos[0], a.pos[1], 'bo')
            plt.plot(self.target[0], self.target[1], 'r+')

    def triangulation(self):
        min_value = 0
        max_value = 100
        for agent in self.schedule.agents:
            signals_for_agent = []
            for senc in self.sensors_arr:
                normalized_signal = (senc.array_of_signals[agent.unique_id][-1] - min_value) / (max_value - min_value)
                signals_for_agent.append(normalized_signal)






    # фильтр частиц
    def particle_filter(self):
        particle_weights = []
        particles_x = []
        particles_y = []

        # Инициализация частиц (собираем координаты агентов в массивы)
        for agent in self.schedule.agents:
            particle_x = agent.pos[0]
            particle_y = agent.pos[1]
            particles_x.append(particle_x)
            particles_y.append(particle_y)
            particle_weights.append(1)

        # Считаем идеальную точку, как среднюю
        ideal_p_x = np.mean(particles_x)
        ideal_p_y = np.mean(particles_y)

        # Обновляем вес частиц, основываясь на расстоянии до идеальной точки
        for step in range(len(particles_x)):
            # Считаем Евклидово расстояние между частицей и идеальной точкой
            distance = np.sqrt((particles_x[step] - ideal_p_x) ** 2 + (particles_y[step] - ideal_p_y) ** 2)
            particle_weights[step] = 1 / distance if distance != 0 else float('inf')

        # Нормализуем веса частиц
        weight_sum = sum(particle_weights)
        particle_weights = [weight / weight_sum for weight in particle_weights]

        # Resample particles based on weights
        new_particles = random.choices(self.schedule.agents, particle_weights, k=num_particles)

        # Удаление агентов, не попавших в выборку
        agents_to_remove = []  # Список для агентов, которые нужно удалить
        for agent in self.schedule.agents:
            if agent not in new_particles:
                agents_to_remove.append(agent)  # Добавление агента в список для удаления

        new_particles_id = []  # айди агентов, которые остались после сортировки( могут совпадать)

        # заполнение массива айдишников агентов после отбора + сортировка
        for i in range(0, len(new_particles)):
            particle = new_particles[i].unique_id
            new_particles_id.append(particle)
        new_particles_id = sorted(new_particles_id)  # Тут лежат отсортированные айдишники агентов, которых мы оставляем

        # Удаление агентов из планировщика
        for agent in agents_to_remove:
            self.schedule.remove(agent)

        new_particles = sorted(new_particles, key=lambda obj: obj.unique_id)
        # пробегаем по айди оставшихся агентов и если айди появляется несколько раз вместо него создаем нового
        k = 0
        # print(*new_particles_id)
        for i in range(1, len(new_particles_id)):
            if new_particles_id[i - 1] == new_particles_id[i]:
                speed = new_particles[i].speed_agent() + random.uniform(-0.3, 0.3)  # берем speed агента клона (ГРАФИК)
                new_agent = PhysicalAgent(self.number_ag, self, speed)
                self.schedule.add(new_agent)
                self.space.place_agent(new_agent, new_particles[i].pos)  # ставим позицию агента клона (ГРАФИК)
                new_agent.reset_waypoints(new_particles[i].waypoints)
                new_agent.pos = new_particles[i].pos
                new_agent.next_waypoint_index = new_particles[i].next_waypoint_index
                self.number_ag += 1

