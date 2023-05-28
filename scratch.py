a = ([[82.6356901331763], [191.286078339731]])

x, y = a[0][0], a[1][0]
print(x, y)

x, y = sympy.symbols("x y", real=True)

eq1 = sympy.Eq((sensors_coordinates[0][0] - x) ** 2 + (sensors_coordinates[0][1] - y) ** 2, distances[0] ** 2)
eq2 = sympy.Eq((sensors_coordinates[1][0] - x) ** 2 + (sensors_coordinates[1][1] - y) ** 2, distances[1] ** 2)
eq3 = sympy.Eq((sensors_coordinates[2][0] - x) ** 2 + (sensors_coordinates[2][1] - y) ** 2, distances[2] ** 2)

x_coordinates_f = []
y_coordinates_f = []




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