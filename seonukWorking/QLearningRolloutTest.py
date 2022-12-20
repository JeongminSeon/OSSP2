import numpy as np
import pandas as pd
import random
from collections import defaultdict
from QuoridorEnv import QuoridorEnv
from DumbAgent import DumbAgent

AGENT_1 = 100
AGENT_2 = 200

WALL_ACT = 16

class QAgentQLearning():
    def __init__(self, width=5):
        self.eps = 0.2
        self.alpha = 0.1
        self.width = width
        self.all_wall_states = []
        self.wall_states=[]
        self.wall_states.append(0)
        self.all_wall_states.append(0)
        self.wall_act_size = 0
        self.wall_num = 0
        self.wall_num_prime = 0
        self.q_table = np.load("C:/OSSP2/OSSP2/Seonuk-Working/100000_dumbagent_save.npy")
        

 
    def select_action(self, state, agent_num):
        action_val = self.q_table[state[1][0][0],state[1][0][1],self.wall_num_prime,:]
        action = np.argmax(action_val)

        return action

def main():
    env = QuoridorEnv(width=5, value_mode=1)
    agent = QAgentQLearning(width=5)
    sr=1
    p1win=0
    p2win=0
    p1rsum=0
    for _ in range(1000):
        done = 0
        env.reset()
        agent_1 = env.register_agent() #agent 등록
        agent_2 = env.register_agent()
        dumbagent_2 = DumbAgent(env, agent_2)
        walk=0

        while True: # 한 에피소드가 끝날 때 까지
            #agent1 가능한 action 선택
            a_1 = agent.select_action(env.get_state(agent_1), agent_1)

            if a_1 in env.get_legal_action(env.get_state(agent_1)):
                s_prime_1, r_1, done = env.step(agent_1, a_1)
                walk+=1
            else:
                while True:
                    a_1 = random.randint(0,35)
                    if a_1 in env.get_legal_action(env.get_state(agent_1)):             
                        break
                s_prime_1, r_1, done = env.step(agent_1, a_1)
                walk+=1

            if done == AGENT_1:
                p1win += 1
                print("episode :", sr, "The number of step : ", walk, "result : player 승리")
                break
        
            a_2 = dumbagent_2.get_action()

            s_prime_2, r_2, done = env.step(agent_2,a_2)
            walk+=1

            if done == AGENT_2:
                p2win += 1
                print("episode :", sr, "The number of step : ", walk, "result : agent 승리")
                break
        
        #agent.anneal_eps()
        sr+=1
    print("p1승리 : ", p1win, "| p2승리", p2win, "| p1 reward 합 : ",p1rsum)

if __name__ == '__main__':
    main()
