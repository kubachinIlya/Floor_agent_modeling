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
from sklearn.linear_model import LinearRegression

# Задаем количество агентов, соответственно количество частиц (это число нужно задать и в самой модели)
number_of_agents = num_particles = 10


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

        TARGET_SENSITIVITY = 5
        search_target = True
        while search_target:
            dx = self.waypoints[self.next_waypoint_index][0] - self.pos[0]
            dy = self.waypoints[self.next_waypoint_index][1] - self.pos[1]
            # dx = ((dx * (math.cos(math.radians(randint(-15, 15))))) + (dy * (math.sin(math.radians(randint(-15, 15))))))  # Отклонение вектора
            # dy = ((dx * (-math.sin(math.radians(randint(-15, 15))))) + (dy * (math.cos(math.radians(randint(-15, 15))))))  # ГРАФИК
            dx = ((dx * (math.cos(math.radians(int(random.gauss(0, 15)))))) + (
                        dy * (math.sin(math.radians(int(random.gauss(0, 15)))))))  # Отклонение вектора
            dy = ((dx * (-math.sin(math.radians(int(random.gauss(0, 15)))))) + (
                        dy * (math.cos(math.radians(int(random.gauss(0, 15)))))))

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


class IdealAgent(Agent):
    def __init__(self, unique_id, model):
        self.is_moving = True
        super().__init__(unique_id, model)
        self.speed_of_agent = 2
        self.agent_step_number = 0
        """""
        таргет сенсивити = 7
        при скорости 1 - 538-539 шагов
        при скорости 2 -271 шаг
        """""

    def model_time(self):
        return self.agent_step_number

    def reset_waypoints(self, waypoints=None):
        if waypoints:
            self.waypoints = waypoints
        self.model.space.move_agent(self, self.waypoints[0])
        self.next_waypoint_index = 1
        self.is_moving = True

    def get_points_to_show(self):
        return {'agent': self.pos, 'final_target': self.waypoints[-1],
                'next_target': self.waypoints[self.next_waypoint_index]}

    def speed_agent(self):
        return self.speed_of_agent

    def step(self):
        self.agent_step_number += 1
        TARGET_SENSITIVITY = 7
        search_target = True
        while search_target:
            dx = self.waypoints[self.next_waypoint_index][0] - self.pos[0]
            dy = self.waypoints[self.next_waypoint_index][1] - self.pos[1]
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
        my_file = open(r".\ideal_agent.txt", "a")
        my_file.write(str(self.agent_step_number) + " ")
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
            distance = sqrt((x_starting_point - x_end_point)**2 + (y_starting_point - y_end_point)**2)
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

            level_of_signal_for_array = 0.1

            for i in range(0, steps_between_pos):
                x_starting_point += step_to_go_x
                y_starting_point += step_to_go_y
                if AgentEnvironmentMap.is_wall(map_for_wall, x_starting_point, y_starting_point):
                    distance += 0  # ЗА СТЕНУ ОТНЯЛИ 0.3 от сигнала
                else:
                    level_of_signal_for_array += 0.0  # ЗА ПРОСТРАНСТВО 0.1 от сигнала
            # print(self.unique_id, " ", 1/level_of_signal_for_array, "steps:", steps_between_pos)
            self.array_of_signals[agent.unique_id].append(1/distance)
        # print(*self.array_of_signals)
        # print()


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
        self.number_ag = 10

        with open(agents_json_path) as f:
            agent_json = json.load(f)
        # создание агентов
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
                    speed = random.uniform(3, 7)  # задаем начальную скорость, график
                    a = PhysicalAgent(k, self, speed)
                    self.schedule.add(a)
                    self.space.place_agent(a, waypoints[0])
                    a.reset_waypoints(waypoints)

        # создание идеального агента
        my_file = open(r".\way_points_history_for_agents\agent" + str(99999) + ".txt", "w+")
        my_file.close()
        waypoints = []
        for i, aj in enumerate(agent_json):
            with open(aj['waypoints_path']) as f:
                lns = f.readlines()
                for ln in lns:
                    parts = re.findall('\d+', ln)
                    waypoints.append((int(parts[0].strip()) - hard_dx, int(parts[1].strip()) - hard_dy))
        ideal_ag = IdealAgent(99999, self)
        self.schedule.add(ideal_ag)
        self.space.place_agent(ideal_ag, waypoints[0])
        ideal_ag.reset_waypoints(waypoints)


        # добавлены сенсоры из файла
        self.array = np.load(
            "Src\Src\medicine-data\iBeacon_data\summer_2_floor\points_wifi_2.npy")
        self.sensors_coordinates = []
        for i in range(0, 9):
            x_coord = self.array[i][0]
            y_coord = self.array[i][1]
            self.sensors_coordinates.append([x_coord - hard_dx, y_coord - hard_dy])
            data_datchik = MySensor(i, (x_coord - hard_dx, y_coord - hard_dy))
            self.sensors_arr.append(data_datchik)

        self.data_collector = DataCollector({'moving_agents_num': 'moving_agents_num'},
                                            {'is_moving': 'is_moving', 'x': lambda a: a.pos[0],
                                             'y': lambda a: a.pos[1]})

        self.moving_agents_num = 0
        self.running = True
        self.data_collector.collect(self)
        # агенты для триангуляции, датчиков и фильтра частиц
        self.agents = []

        self.step_number = 0  # шаг модели для вызова фильтра частиц на определенных шагах

    def step(self):
        self.schedule.step()
        # print(self.schedule.time, "это время")
        self.data_collector.collect(self)
        self.step_number += 1
        # print(self.step_number) вывод номера шага модели
        self.moving_agents_num = sum([a.is_moving for a in self.schedule.agents])
        self.running = self.moving_agents_num > 0
        checker_for_moving = False
        self.agents = self.schedule.agents
        for agent in self.agents:
            if agent.unique_id == 99999:
                self.agents.remove(agent)
            if agent.is_moving and agent.unique_id != 99999:
                checker_for_moving = True

        # вызов фильтра частиц на каждом определенном шаге ГРАФИК
        if self.step_number % 7 == 0 and checker_for_moving:
            self.particle_filter(self.agents)
            print('СРАБОТАЛ ФИЛЬТР')

        # вызов сенсоров для работы по определению уровня сигнала ПРОПИСАТЬ ЕСЛИ ЕЩЕ ЕСТЬ АГЕНТЫ
        if checker_for_moving:
            for sens in self.sensors_arr:
                sens.sense_signal(self.agents)
        # вызов триангуляции
        if checker_for_moving:
            if self.step_number > 0:
                self.triangulation()

        self.mean_coord(self.agents)


    def mean_coord(self, agent_for_particles):
        particle_weights = []
        particles_x = []
        particles_y = []
        for agent in agent_for_particles:
            particle_x = agent.pos[0]
            particle_y = agent.pos[1]
            particles_x.append(particle_x)
            particles_y.append(particle_y)
            particle_weights.append(1)

        # Считаем идеальную точку, как среднюю
        # ideal_p_x_a = np.mean(particles_x)
        # ideal_p_y_a = np.mean(particles_y)
        # print(ideal_p_x_a, ideal_p_y_a, "позиция среднее арифметическое(не используется в фильтре)")

        # считаем идеальную точку линейной регрессией
        # Create an array of corresponding indices for the particles
        indices_x = np.arange(len(particles_x)).reshape(-1, 1)
        indices_y = np.arange(len(particles_y)).reshape(-1, 1)
        # Initialize and fit the linear regression model
        regression_model_x = LinearRegression()
        regression_model_y = LinearRegression()
        regression_model_x.fit(indices_x, particles_x)
        regression_model_y.fit(indices_y, particles_y)
        # Predict the values based on the linear regression model
        predicted_p_x = regression_model_x.predict(indices_x)
        predicted_p_y = regression_model_y.predict(indices_y)
        # Calculate the ideal_p_x as the mean of the predicted values
        ideal_p_x = np.mean(predicted_p_x)
        ideal_p_y = np.mean(predicted_p_y)
        my_file = open(r".\mean_position.txt", "a")
        my_file.write(str(ideal_p_x) + " " + str(ideal_p_y) + " ")
        my_file.close()

    def plot_explicitly(self):
        plt.imshow(self.env_map.img)
        for a in self.schedule.agents:
            plt.plot(a.pos[0], a.pos[1], 'bo')
            plt.plot(self.target[0], self.target[1], 'r+')

    def triangulation(self):
        mean_signals = []  # Храним средние значения сигнала каждого сенсора для всех агентов на шаге модели

        # находим среднее значение сигнала каждого сенсора и собираем в mean_signals
        for sens in self.sensors_arr:
            sum_signals = 0
            for i in range(0, len(sens.array_of_signals)):
                sum_signals += sens.array_of_signals[i][-1]
                # print(sens.array_of_signals[i][-1])
            mean_sum = sum_signals / len(sens.array_of_signals)
            mean_signals.append(mean_sum)
        # print(*mean_signals, " сигналы в триангуляции средние")
        # минимальное и максимальное значения сигнала для нормализации типа min - max
        min_value = min(mean_signals)
        max_value = max(mean_signals)

        # нормализуем уровень сигнала range( 0 - 1 )
        for i in range(0, len(mean_signals)):
            normalized_signal = (mean_signals.__getitem__(i) - min_value) / (max_value - min_value)
            # normalized_signal = (mean_signals.__getitem__(i) - min_value) / (max_value - min_value) * 2 - 1
            mean_signals[i] = normalized_signal
        # print(mean_signals, "нормализованные сигналы в триангуляции")
        # готовые массив сигналов сенсоров, массив координат сенсоров для реализации триангуляции
        # Тут можно сделать срез, чтобы получить разное количество датчиков ( ГРАФИК)
        mean_signals = np.array(mean_signals)
        sensors_coordinates = np.array(self.sensors_coordinates)

        agent_position_x = 0
        agent_position_y = 0
        # Нормализуем веса датчиков
        weight_sum = sum(mean_signals)
        sensors_weights = [weight / weight_sum for weight in mean_signals]
        max_indices = sorted(range(len(sensors_weights)), key=lambda i: sensors_weights[i])[-3:]
        max_weights = sorted(sensors_weights)[-3:]

        for i in range(0, 3):
            max_weights[i] = max_weights[i]/(sum(max_weights))
        for i in range(0, 3):
            agent_position_x += max_weights[i] * sensors_coordinates[max_indices[i]][0]
            agent_position_y += max_weights[i] * sensors_coordinates[max_indices[i]][1]
        """""
        for i in range(0, 9):
            agent_position_x += sensors_weights[i] * sensors_coordinates[i][0]
            agent_position_y += sensors_weights[i] * sensors_coordinates[i][1]
        # конец неработающей части
        """""

        print(agent_position_x, agent_position_y, "позиция по триангуляции")
        my_file = open(r".\triangulation.txt", "a")
        my_file.write(str(agent_position_x) + " " + str(agent_position_y) + " ")
        my_file.close()

    # фильтр частиц
    def particle_filter(self, agent_for_particles):
        particle_weights = []
        particles_x = []
        particles_y = []

        for agent in agent_for_particles:
            particle_x = agent.pos[0]
            particle_y = agent.pos[1]
            particles_x.append(particle_x)
            particles_y.append(particle_y)
            particle_weights.append(1)

        # Считаем идеальную точку, как среднюю
        # ideal_p_x_a = np.mean(particles_x)
        # ideal_p_y_a = np.mean(particles_y)
        # print(ideal_p_x_a, ideal_p_y_a, "позиция среднее арифметическое(не используется в фильтре)")

        # считаем идеальную точку линейной регрессией
        # Create an array of corresponding indices for the particles
        indices_x = np.arange(len(particles_x)).reshape(-1, 1)
        indices_y = np.arange(len(particles_y)).reshape(-1, 1)
        # Initialize and fit the linear regression model
        regression_model_x = LinearRegression()
        regression_model_y = LinearRegression()
        regression_model_x.fit(indices_x, particles_x)
        regression_model_y.fit(indices_y, particles_y)
        # Predict the values based on the linear regression model
        predicted_p_x = regression_model_x.predict(indices_x)
        predicted_p_y = regression_model_y.predict(indices_y)
        # Calculate the ideal_p_x as the mean of the predicted values
        ideal_p_x = np.mean(predicted_p_x)
        ideal_p_y = np.mean(predicted_p_y)

        print(ideal_p_x, " ", ideal_p_y, " позиция с помощью линейной регрессии")
        # Обновляем вес частиц, основываясь на расстоянии до идеальной точки
        for step in range(len(particles_x)):
            # Считаем Евклидово расстояние между частицей и идеальной точкой
            distance = np.sqrt((particles_x[step] - ideal_p_x) ** 2 + (particles_y[step] - ideal_p_y) ** 2)
            particle_weights[step] = 1 / distance if distance != 0 else float('100')

        # Нормализуем веса частиц
        weight_sum = sum(particle_weights)
        particle_weights = [weight / weight_sum for weight in particle_weights]

        # тут выбраны агенты для клонирования
        new_particles = random.choices(agent_for_particles, particle_weights, k=num_particles)

        # Удаление агентов, не попавших в выборку
        agents_to_remove = []  # Список для агентов, которые нужно удалить
        for agent in self.schedule.agents:
            if agent not in new_particles and agent.unique_id != 99999:
                agents_to_remove.append(agent)  # Добавление агента в список для удаления

        new_particles_id = []  # айди агентов для клонирования

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
        counter_for_plot = 0
        for i in range(1, len(new_particles_id)):
            if new_particles_id[i - 1] == new_particles_id[i]:
                counter_for_plot += 1
                speed = new_particles[i].speed_agent() + random.uniform(-0.3, 0.3)  # берем speed агента клона (ГРАФИК)
                new_agent = PhysicalAgent(self.number_ag, self, speed)
                self.schedule.add(new_agent)
                self.space.place_agent(new_agent, new_particles[i].pos)  # ставим позицию агента клона (ГРАФИК)
                new_agent.reset_waypoints(new_particles[i].waypoints)
                new_agent.pos = new_particles[i].pos
                new_agent.next_waypoint_index = new_particles[i].next_waypoint_index
                self.number_ag += 1

        my_file = open(r".\counter_new_waves.txt", "a")
        my_file.write(str(counter_for_plot) + " ")
        my_file.close()

