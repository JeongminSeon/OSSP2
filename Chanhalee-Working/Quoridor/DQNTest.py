import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from QuoridorEnv import QuoridorEnv

WIDTH = 5


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(6 + ((WIDTH-1)*(WIDTH-1)*2), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 4 + ((WIDTH-1)*(WIDTH-1)*2))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


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


def main():
    env = QuoridorEnv(width=WIDTH, value_mode=0)
    q = Qnet()
    q.load_state_dict(torch.load(
        'woocheol\Quoridor\save_models\Quoridor_DQN_w5_200000.pth', map_location='cpu'))

    env.reset()
    done = False

    while True:
        agent_num = 300 - env.last_played
        original_state = env.get_state(agent_num)
        state = getLinearState(original_state)

        available_actions = env.get_legal_action(original_state)

        env.render(agent_num)

        if (agent_num == 100):
            print(available_actions)
            action = int(input())
            env.step(agent_num, action)
            continue

        out = q.forward(torch.from_numpy(np.array(state)).float())
        show_list = []

        for i in range(4 + ((WIDTH-1)*(WIDTH-1)*2)):
            if i in available_actions:
                show_list.append(round(out[i].item(), 4))
            else:
                show_list.append(round(out[i].item(), 4))

        action = show_list.index(max(show_list))
        print(action)
        env.step(agent_num, action)


main()
