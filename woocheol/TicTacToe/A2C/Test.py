from TicTacToe import TicTacToe
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.data = []

        self.v_fc1 = nn.Linear(9*3, 256)
        self.v_fc2 = nn.Linear(256, 256)
        self.v_fc3 = nn.Linear(256, 256)
        self.v_fc4 = nn.Linear(256, 256)
        self.v_fc5 = nn.Linear(256, 1)

        self.pi_fc1 = nn.Linear(9*3, 256)
        self.pi_fc2 = nn.Linear(256, 256)
        self.pi_fc3 = nn.Linear(256, 256)
        self.pi_fc4 = nn.Linear(256, 256)
        self.pi_fc5 = nn.Linear(256, 9)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.pi_fc1(x))
        x = F.relu(self.pi_fc2(x))
        x = F.relu(self.pi_fc3(x))
        x = F.relu(self.pi_fc4(x))
        x = self.pi_fc5(x)

        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.v_fc1(x))
        x = F.relu(self.v_fc2(x))
        x = F.relu(self.v_fc3(x))
        x = F.relu(self.v_fc4(x))
        v = self.v_fc5(x)
        return v


env = TicTacToe()

q = Qnet()
q.load_state_dict(torch.load('woocheol/A2C/A2C.pth', map_location='cpu'))

ai_win = 0

for t in range(1000):
    s = env.reset()
    ai_turn = 0
    while True:
        env.render()

        if ai_turn == 0:
            action = random.choice(env.available_actions())
            s_prime, r, done = env.step(action)
            s = s_prime
        else:
            obs = q.pi(torch.from_numpy(np.array(s * 3)).float())
            print(obs)

            available = env.available_actions()
            show_list = []
            for i in range(9):
                if i in available:
                    show_list.append(obs[i].item())
                else:
                    show_list.append(0)

            a = show_list.index(max(show_list))
            print(show_list, a)
            a = int(input())
            s_prime, r, done = env.step(a)
        print(s_prime, r, done)

        if done:
            env.render()
            if (ai_turn == 1):
                if r == 1:
                    print("AI Tie!")
                    ai_win += 1
                if r == 2:
                    print("AI Win!")
                    ai_win += 1
            else:
                if r == 1:
                    print("AI Tie!")
                    ai_win += 1
                if r == 2:
                    print("AI Lose!")
            break

        ai_turn = 1 - ai_turn
        s = s_prime

print("AI Win Rate: ", ai_win / 1000)
