import os
import base64
from mesa.visualization.ModularVisualization import ModularServer, VisualizationElement
from mesa.visualization.modules import TextElement
from mesa.visualization.UserParam import UserSettableParameter
from Agents import IndoorModel


class IndoorVisualCanvas(VisualizationElement):
    local_includes = ["simple_continuous_canvas.js"]

    def __init__(self, portrayal_method, canvas_height=1500, canvas_width=1500, bg_path=None):
        super().__init__()
        self.portrayal_method = portrayal_method
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        new_element = f"new Simple_Continuous_Module({self.canvas_width}, {self.canvas_height}, {bg_path})"
        self.js_code = "elements.push(" + new_element + ");"

    def transform_xy(self, model, pos):
        x, y = pos
        x = (x - model.space.x_min) / (model.space.x_max - model.space.x_min)
        y = (y - model.space.y_min) / (model.space.y_max - model.space.y_min)
        return x, y

    def render(self, model):
        space_state = []
        for obj in model.schedule.agents:
            # agent
            portrayal = self.portrayal_method(obj)
            portrayal["x"], portrayal["y"] = self.transform_xy(model, obj.pos)
            space_state.append(portrayal)
            # targets
            tg = obj.get_points_to_show()
            tm_portrayal = {'Shape': 'circle', 'r': 2, 'Filled': 'true', 'Color': 'magenta'}
            tm_portrayal["x"], tm_portrayal["y"] = self.transform_xy(model, tg['next_target'])
            space_state.append(tm_portrayal)
            tf_portrayal = {'Shape': 'circle', 'r': 2, 'Filled': 'true', 'Color': 'red'}
            tf_portrayal["x"], tf_portrayal["y"] = self.transform_xy(model, tg['final_target'])
            space_state.append(tf_portrayal)
        return space_state


class RunningAgentsNum(TextElement):
    def __init__(self):
        super().__init__()

    def render(self, model):
        return f'Running agents: {model.moving_agents_num}'


def agent_portrayal(a):
    cl = '#00FF00' if a.is_moving else '#0000FF'
    return {'Shape': 'circle', 'r': 4, 'Filled': 'true', 'Color': cl}


bg_path = 'map_2floor_bw.png'
with open(bg_path, "rb") as img_file:
    b64_string = base64.b64encode(img_file.read())

running_counter_element = RunningAgentsNum()
canvas_element = IndoorVisualCanvas(agent_portrayal, 250, 600, 'data:image/png;base64, ' + b64_string.decode('utf-8'))


model_params = {
    # 'N': UserSettableParameter('slider', 'N', 5, 1, 20),
    'agents_json_path': 'agents.json',
    'env_map_path': bg_path
}

server = ModularServer(IndoorModel, [canvas_element, running_counter_element], 'Indoor model', model_params)

