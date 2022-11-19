import numpy as np
import random
from collections import defaultdict
from QuoridorEnv import QuoridorEnv

AGENT_1 = 100
AGENT_2 = 200


class QAgentQLearning():
    def __init__(self, width=5):
        self.q_table_1 = np.zeros((width, width, 4 + (width-1) * (width-1) * 2)) # agent1의 q벨류를 저장하는 변수. 모두 0으로 초기화. 
        self.q_table_2 = np.zeros((width, width, 4 + (width-1) * (width-1) * 2)) # agent2의 q벨류를 저장하는 변수. 모두 0으로 초기화. 
        self.eps = 0.9 
        self.alpha = 0.1
        self.width = width
        
    def select_action(self, state, agent_num):
        # eps-greedy로 액션을 선택
        coin = random.random()
        if (agent_num != AGENT_1):
            if coin < self.eps:
                action = random.randint(0,3 + (self.width-1) * (self.width-1) * 2)
            else:
                action_val = self.q_table_2[state[1][1][0],state[1][1][1],:]
                action = np.argmax(action_val)
            return action
        else:
            if coin < self.eps:
                action = random.randint(0,3 + (self.width-1) * (self.width-1) * 2)
            else:
                action_val = self.q_table_1[state[1][0][0],state[1][0][1],:]
                action = np.argmax(action_val)
            return action

    def update_table(self, state, a, r, s_prime, agent_num):
        # 한 에피소드에 해당하는 history를 입력으로 받아 q 테이블의 값을 업데이트 한다
        if (agent_num != AGENT_1):
            #QLearning 업데이트 식을 이용
            self.q_table_2[state[1][1][0],state[1][1][1],a] = self.q_table_2[state[1][1][0],state[1][1][1],a] + self.alpha * (r + np.amax(self.q_table_2[s_prime[1][1][0],s_prime[1][1][1],:]) - self.q_table_2[state[1][1][0],state[1][1][1],a])
            
        else:
            
            #QLearning 업데이트 식을 이용
            self.q_table_1[state[1][0][0],state[1][0][1],a] = self.q_table_1[state[1][0][0],state[1][0][1],a] + self.alpha * (r + np.amax(self.q_table_1[s_prime[1][0][0],s_prime[1][0][1],:]) - self.q_table_1[state[1][0][0],state[1][0][1],a])
            
        #print(self.q_table_1)

    def anneal_eps(self):
        self.eps -= 0.01
        self.eps = max(self.eps, 0.2)

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
    agent = QAgentQLearning(width=5)
    sr=1
    

    for _ in range(1000): # 총 1,000 에피소드 동안 학습
        done = 0
        env.reset()
        agent_1 = env.register_agent() #agent 등록
        agent_2 = env.register_agent()
        walk=1

        #env.render(agent_1)
        while done == 0: # 한 에피소드가 끝날 때 까지
            while True: 
                a_1 = agent.select_action(env.get_state(agent_1), agent_1) #agent1 가능한 action 선택
                
                if a_1 in env.get_legal_action(env.get_state(agent_1)):                
                    break
            
            s_prime_1, r_1, done = env.step(agent_1, a_1) #action 진행 후 state, reward, done 반환
            agent.update_table(env.get_state(agent_1),a_1,r_1,s_prime_1, agent_1)
            #print(env.get_state(agent_1))
            #env.render(agent_1)
            walk+=1

            if done == AGENT_1:
                break
            while True: #agent2 가능한 action 선택
                a_2 = agent.select_action(env.get_state(agent_2), agent_2)

                if a_2 in env.get_legal_action(env.get_state(agent_2)):
                    break

            s_prime_2, r_2, done = env.step(agent_2,a_2)
            agent.update_table(env.get_state(agent_2),a_2,r_2,s_prime_2, agent_2)
            #print(env.get_state(agent_1))
            #env.render(agent_1)
        if done == AGENT_1:
            #print("finished at", s)
            print("episode :", sr, "The number of step : ", walk, "result : p1승리")
        elif done == AGENT_2:
            #print("finished at", s)
            print("episode :", sr, "The number of step : ", walk, "result : p2승리")
        sr+=1

        #env.render(agent_1)
        agent.anneal_eps()

    agent.show_table() # 학습이 끝난 결과를 출력

if __name__ == '__main__':
    main()