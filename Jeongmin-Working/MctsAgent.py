from QuoridorEnv import *
from mcts_quoridor import *

class Agent():
    def __init__(self, env, agent_num) :
        self.agent_num = agent_num
        self.env = env

    def get_action(self) :
        if self.agent_num != self.env.get_last_played():
            
            self.env.render(100)
            mcts = MCTS()
            best_action = mcts.search(self.env)

            return best_action.get_action()

