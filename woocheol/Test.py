from TicTacToe import TicTacToe
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import collections
import random


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 9)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()


env = TicTacToe()

q = Qnet()
q.load_state_dict(torch.load('DQN1.pth', map_location='cpu'))

ai_win = 0

for t in range(100):
    s = env.reset()
    ai_turn = 1
    while True:
        # env.render()

        if ai_turn == 0:
            action = random.choice(env.available_actions())
            s_prime, r, done = env.step(action)
            s = s_prime
        else:
            obs = q.forward(torch.from_numpy(np.array(s)).float())

            available = env.available_actions()
            show_list = []
            for i in range(9):
                if i in available:
                    show_list.append(round(obs[i].item(), 4))
                else:
                    show_list.append(0)

            a = show_list.index(max(show_list))
            s_prime, r, done = env.step(a)

        if done:
            if (ai_turn == 1 and r == 1):
                env.render()
                print("AI Win!")
                ai_win += 1
                print("AI Win Rate: ", ai_win, t)
            else:
                print("AI Lose!")
            break

        ai_turn = 1 - ai_turn
        s = s_prime
