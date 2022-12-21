import random
import pygame
import time
import sys
from pygame.locals import *
from ChanhaleeWorking.QuoridorEnv import QuoridorEnv
from ChanhaleeWorking.QuoridorGUI import QuoridorGUI
from JeongminWorking.MctsAgent import Agent


env = QuoridorEnv(width=5, value_mode=9)
agent_1 = env.register_agent()
agent_2 = env.register_agent()
mcts = Agent(env, agent_2)
# g = QuoridorGUI(q, agent_num1=agent_1, agent_num2=agent_2,
#                 auto_pilot1=None, auto_pilot2=None)
g = QuoridorGUI(env, agent_num1=agent_1,
                auto_pilot1=mcts, auto_pilot2=None)
# g = QuoridorGUI(q, auto_pilot1=dumb, auto_pilot2=dumb2)
g.startGame()
