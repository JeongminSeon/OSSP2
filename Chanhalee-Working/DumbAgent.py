import numpy as np
from Quoridor import QuoridorEnv
import random


class DumbAgent():
    def __init__(self, env, agent_num):
        self.agent_num = agent_num
        self.env = env

    def get_action(self):
        if self.agent_num != self.env.get_last_played():
            move = random.choice(self.env.get_legal_action(
                self.env.get_state(self.agent_num)))
            print("randM: ", end="")
            print(move)
            return move
        else:
            print("randM: not My turn")
            return None

    def get_agent_num(self):
        return self.agent_num
