import random
import pygame
import time
import sys
import numpy as np
from pygame.locals import *
<<<<<<< HEAD:Heuristic_Agent.py
from ChanhaleeWorking.QuoridorEnv import QuoridorEnv
from ChanhaleeWorking.QuoridorGUI import QuoridorGUI
<<<<<<<< HEAD:PlayWithHeuristicAgent.py
from ChanhaleeWorking.HeuristicAgent import HeuristicAgent


env = QuoridorEnv(width=9, value_mode=9)
agent_1 = env.register_agent()
agent_2 = env.register_agent()
dumb = HeuristicAgent(env, agent_2)
dumb2 = HeuristicAgent(env, agent_1)
print(env.get_legal_action(env.get_state(agent_1)))
env.render(agent_1)
# g = QuoridorGUI(q, agent_num1=agent_1, agent_num2=agent_2,
#                 auto_pilot1=None, auto_pilot2=None)
g = QuoridorGUI(env, agent_num1=agent_1,
                auto_pilot1=dumb, auto_pilot2=None)
========
from ChanhaleeWorking.DumbAgent import DumbAgent
=======
from QuoridorEnv import QuoridorEnv
#from DumbAgent import DumbAgent
from QAgentQLearning import QAgentQLearning

>>>>>>> 365710c66b4560fcca3c4596a77c9d1478938263:Seonuk-Working/Executable/Heuristic&QLearning/QuoridorGUI.py


q = QuoridorEnv(width=5, value_mode=1)
agent_1 = q.register_agent()
agent_2 = q.register_agent()
q_agent = QAgentQLearning(q, agent_2)
#dumb2 = DumbAgent(q, agent_1)
print(q.get_legal_action(q.get_state(agent_1)))
q.render(agent_1)
# g = QuoridorGUI(q, agent_num1=agent_1, agent_num2=agent_2,
#                 auto_pilot1=None, auto_pilot2=None)
g = QuoridorGUI(q, agent_num1=agent_1,
                auto_pilot1=q_agent, auto_pilot2=None)
>>>>>>>> refs/remotes/origin/main:Seonuk-Working/Executable/Heuristic&QLearning/QuoridorGUI.py
# g = QuoridorGUI(q, auto_pilot1=dumb, auto_pilot2=dumb2)
g.startGame()