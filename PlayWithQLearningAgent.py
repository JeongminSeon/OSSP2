from SeonukWorking.QAgentQLearning import QAgentQLearning
from ChanhaleeWorking.QuoridorGUI import QuoridorGUI
from ChanhaleeWorking.QuoridorEnv import QuoridorEnv
import random
import pygame
import time
import sys
import numpy as np
from pygame.locals import *


env = QuoridorEnv(width=5, value_mode=9)
agent_1 = env.register_agent()
agent_2 = env.register_agent()
ql = QAgentQLearning(env, agent_2)
# g = QuoridorGUI(q, agent_num1=agent_1, agent_num2=agent_2,
#                 auto_pilot1=None, auto_pilot2=None)
g = QuoridorGUI(env, agent_num1=agent_1,
                auto_pilot1=ql, auto_pilot2=None)
# g = QuoridorGUI(q, auto_pilot1=dumb, auto_pilot2=dumb2)
g.startGame()
