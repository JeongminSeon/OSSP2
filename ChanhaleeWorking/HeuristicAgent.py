import numpy as np
import random

ACT_MOVE_CNT = 4
ACT_MOVE_NORTH = 0
ACT_MOVE_WEST = 1
ACT_MOVE_SOUTH = 2
ACT_MOVE_EAST = 3

# 휴리스틱으로만 움직이는 agent


class HeuristicAgent():
    def __init__(self, env, agent_num):
        self.agent_num = agent_num
        self.env = env
        self.wall_prob = 0.5

    def get_action(self, state):
        if self.agent_num != self.env.get_last_played():
            max_val_w = -99999
            max_val_m = -99999
            max_action_w = []
            max_action_m = []
            moves = self.env.get_legal_action(
                state)

            for action in moves:
                s_state = state
                state_temp = (s_state[0].copy(), s_state[1].copy())
                width = self.env.width
                reward = 0
                if (action < 4):
                    if (action == ACT_MOVE_NORTH):
                        state_temp[1][0][1] += 1
                    if (action == ACT_MOVE_WEST):
                        state_temp[1][0][0] -= 1
                    if (action == ACT_MOVE_SOUTH):
                        state_temp[1][0][1] -= 1
                    if (action == ACT_MOVE_EAST):
                        state_temp[1][0][0] += 1
                elif (action < self.env.all_action.size):
                    state_temp[1][0][2] -= 1
                    action -= ACT_MOVE_CNT
                    col_row = action // ((width - 1) * (width - 1))
                    pos_x = (action %
                             ((width - 1) * (width - 1))) % (width - 1)
                    pos_y = (action %
                             ((width - 1) * (width - 1))) // (width - 1)
                    state_temp[0][col_row][pos_x][pos_y] = True
                    action += ACT_MOVE_CNT
                isItEnd = self.env.ask_end_state(state_temp)
                if (isItEnd == 0):
                    if (state_temp[1][1][1] == 1):  # 상대방의 승리 직전
                        reward = -1000
                    reward = self.env.ask_how_far_opp(
                        state_temp) - self.env.ask_how_far(state_temp) * 2 - 1
                elif (state_temp[1][0][1] == width - 1):
                    reward = 1000
                else:
                    reward = -1000

                if action < ACT_MOVE_CNT:
                    if max_val_m <= reward:
                        if reward == 1000:
                            print(action)
                            return action
                        elif max_val_m < reward:
                            max_action_m = []
                        max_action_m.append(action)
                        max_val_m = reward
                else:
                    if max_val_w <= reward:
                        if reward == 1000:
                            return action
                        elif max_val_w < reward:
                            max_action_w = []
                        max_action_w.append(action)
                        max_val_w = reward

            print(max_action_w, max_action_m)
            if len(max_action_w) > 0:
                if random.random() < self.wall_prob:
                    return random.choice(max_action_w)

            return random.choice(max_action_m)
        else:
            print("randM: not My turn")
            return None

    def get_agent_num(self):
        return self.agent_num
