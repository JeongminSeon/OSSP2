import numpy as np
import random
from collections import defaultdict
from QuoridorEnv import QuoridorEnv

AGENT_1 = 100
AGENT_2 = 200


class QAgent():
    def __init__(self, width=5):
        self.q_table_1 = np.zeros((width, width, 4 + (width-1) * (width-1) * 2+1)) # agent1의 q벨류를 저장하는 변수. 모두 0으로 초기화. 
        self.q_table_2 = np.zeros((width, width, 4 + (width-1) * (width-1) * 2+1)) # agent2의 q벨류를 저장하는 변수. 모두 0으로 초기화. 
        self.eps = 0.9 
        self.alpha = 0.01
        self.width = width
        
    def select_action(self, state, agent_num):
        # eps-greedy로 액션을 선택
        coin = random.random()
        if (agent_num != AGENT_1):
            if coin < self.eps:
                action = random.randint(0,4 + (self.width-1) * (self.width-1) * 2)
            else:
                action_val = self.q_table_2[state[1][1][0],state[1][1][1],:]
                action = np.argmax(action_val)
            return action
        else:
            if coin < self.eps:
                action = random.randint(0,4 + (self.width-1) * (self.width-1) * 2)
            else:
                action_val = self.q_table_1[state[1][0][0],state[1][0][1],:]
                action = np.argmax(action_val)
            return action

    def update_table(self, history, agent_num):
        # 한 에피소드에 해당하는 history를 입력으로 받아 q 테이블의 값을 업데이트 한다
        cum_reward_1 = 0
        cum_reward_2 = 0
        if (agent_num != AGENT_1):
            for transition in history[::-1]:
                state, a, r, s_prime = transition
                # 몬테 카를로 방식을 이용하여 업데이트.
                self.q_table_2[state[1][1][0],state[1][1][1],a] = self.q_table_2[state[1][1][0],state[1][1][1],a] 
                + self.alpha * (cum_reward_2 - self.q_table_2[state[1][1][0],state[1][1][1],a])
                cum_reward_2 = cum_reward_2 + r
        else:
            for transition in history[::-1]:
                state, a, r, s_prime = transition
                # 몬테 카를로 방식을 이용하여 업데이트.
                self.q_table_1[state[1][0][0],state[1][0][1],a] = self.q_table_1[state[1][0][0],state[1][0][1],a] 
                + self.alpha * (cum_reward_1 - self.q_table_1[state[1][0][0],state[1][0][1],a])
                cum_reward_1 = cum_reward_1 + r

    def anneal_eps(self):
        self.eps -= 0.03
        self.eps = max(self.eps, 0.1)

    def show_table(self):
        # 학습이 각 위치에서 어느 액션의 q 값이 가장 높았는지 보여주는 함수
        q_lst = self.q_table_1.tolist()
        data = np.zeros((self.width,self.width))
        for row_idx in range(len(q_lst)):
            row = q_lst[row_idx]
            for col_idx in range(len(row)):
                col = row[col_idx]
                action = np.argmax(col)
                data[row_idx, col_idx] = action
        print(data)
      
def main():
    env = QuoridorEnv(width=5, value_mode=1)
    agent = QAgent(width=5)
    sr=1

    for _ in range(1000): # 총 1,000 에피소드 동안 학습
        done = 0
        history_1 = []
        history_2 = []
        agent_1 = env.register_agent()
        agent_2 = env.register_agent()

        env.reset()
        s_1 = env.get_state(agent_1)
        s_2 = env.get_state(agent_2)
        while done == 0: # 한 에피소드가 끝날 때 까지
            while True: #agent1 가능한 action 선택
                a_1 = agent.select_action(s_1, agent_1)
                if a_1 in env.get_legal_action(s_1):
                    break
            s_prime_1, r_1, done = env.step(agent_1, a_1)
            history_1.append((s_1, a_1, r_1, s_prime_1))
            s_1 = s_prime_1
            while True: #agent2 가능한 action 선택
                a_2 = agent.select_action(s_2, agent_2)
                if a_2 in env.get_legal_action(s_2):
                    break
            s_prime_2, r_2, done = env.step(agent_2,a_2)
            history_2.append((s_2, a_2, r_2, s_prime_2))
            s_2 = s_prime_1
        if done == AGENT_1:
            print(sr, ": p1승리")
        elif done == AGENT_2:
            print(sr, ": p2승리")
        sr+=1


        agent.update_table(history_1, agent_1) # 히스토리를 이용하여 에이전트를 업데이트
        agent.update_table(history_2, agent_2)
        agent.anneal_eps()

    agent.show_table() # 학습이 끝난 결과를 출력

if __name__ == '__main__':
    main()