import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from QuoridorEnv import QuoridorEnv

learning_rate = 0.0005
gamma = 1
buffer_limit = 100000
batch_size = 32

WIDTH = 5


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
            torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
            torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(6 + ((WIDTH-1)*(WIDTH-1)*2), 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 4 + ((WIDTH-1)*(WIDTH-1)*2))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

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
                out_list = out.tolist()
                new_out_list = []
                for i in range(len(out_list)):
                    if i not in available_actions:
                        new_out_list.append(-100)
                    else:
                        new_out_list.append(out_list[i])
                return new_out_list.index(max(new_out_list))


def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


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
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 20
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(1):
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200)
                      )  # Linear annealing from 8% to 1%
        state = getLinearState(env.reset())
        done = False

        while not done:
            agent_turn = 300 - env.last_played

            env.render(agent_turn)
            available_actions = env.get_legal_action()
            action = q.sample_action(torch.tensor(
                state).float(), epsilon, available_actions)

            print(available_actions)
            print(action)
            state_prime, reward, done = env.step(agent_turn, action)
            state_prime = getLinearState(state_prime)

            done_mask = 0.0 if done else 1.0
            memory.put((state, action, reward, state_prime, done_mask))
            state = state_prime

            if done:
                print("Done")
                break

        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print("# of episode :{}, avg score : {:.1f}".format(
                n_epi, reward))
            torch.save(q.state_dict(), f'./save_model/Quoridor{n_epi}.pth')


main()
