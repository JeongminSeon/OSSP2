import random
import pygame
import time
import sys
from pygame.locals import *
from ChanhaleeWorking.QuoridorEnv import QuoridorEnv
from ChanhaleeWorking.QuoridorGUI import QuoridorGUI
from ChanhaleeWorking.DumbAgent import DumbAgent


q = QuoridorEnv(width=9, value_mode=9)
agent_1 = q.register_agent()
agent_2 = q.register_agent()
dumb = DumbAgent(q, agent_2)
dumb2 = DumbAgent(q, agent_1)
print(q.get_legal_action(q.get_state(agent_1)))
q.render(agent_1)
# g = QuoridorGUI(q, agent_num1=agent_1, agent_num2=agent_2,
#                 auto_pilot1=None, auto_pilot2=None)
g = QuoridorGUI(q, agent_num1=agent_1,
                auto_pilot1=dumb, auto_pilot2=None)
# g = QuoridorGUI(q, auto_pilot1=dumb, auto_pilot2=dumb2)
g.startGame()
