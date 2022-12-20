import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class Qnet(nn.Module):
    def __init__(self, env, WIDTH, agent_num):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(6 + ((WIDTH-1)*(WIDTH-1)*2), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 4 + ((WIDTH-1)*(WIDTH-1)*2))

        self.env = env
        self.load_state_dict(
            torch.load(
                'woocheol\Quoridor\save_models\Quoridor_DQN_w5_200000.pth', map_location='cpu')
        )

        self.agent_num = agent_num

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

    def get_action(self):
        state = self.env.get_state(self.agent_num)
        linear_state = torch.tensor(getLinearState(state)).float()
        # print(state, linear_state)
        out = self.forward(linear_state)
        return out.argmax().item()

    def get_agent_num(self):
        return self.agent_num

    def sample_action(self, obs, epsilon, available_actions):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.choice(available_actions)
        else:
            action = out.argmax().item()
            if action in available_actions:
                return action
            else:
                return random.choice(available_actions)


def getLinearState(state):
    map_state, player_status = state

    ret_list = []

    ret_list.append(player_status[0][0])
    ret_list.append(player_status[0][1])
    ret_list.append(player_status[0][2])
    ret_list.append(player_status[1][0])
    ret_list.append(player_status[1][1])
    ret_list.append(player_status[1][2])

    for i in map_state:
        for j in i:
            for k in j:
                ret_list.append(k)
    return ret_list
