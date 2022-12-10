import numpy as np
import random
from collections import defaultdict
from QuoridorEnv import QuoridorEnv

AGENT_1 = 100
AGENT_2 = 200
WALL_ACT = 16


class QAgentQLearning():
    def __init__(self, width=5):
        self.eps = 0.9 
        self.alpha = 0.1
        self.width = width
        self.all_wall_states = []
        self.wall_states=[]
        self.wall_states.append(0)
        self.all_wall_states.append(0)
        self.wall_act_size = 0
        self.wall_num = 0
        self.wall_num_prime = 0

        #벽이 1개 일 때
        for a in range(4, 20):
            self.all_wall_states.append(a)
            self.wall_act_size += 1
        
        for a in range(20, 36):
            self.all_wall_states.append(a)
            self.wall_act_size += 1

        #벽이 2개 일 때
        for a in range(4, 20):
            if a%4 == 3:
                for b in range(a+1,20):
                    self.all_wall_states.append((a, b))
                    self.wall_act_size += 1
            else:
                for b in range(a+2, 20):
                    self.all_wall_states.append((a, b))
                    self.wall_act_size += 1

        for a in range(20, 36):
            for b in range(a+1, 36):
                if b != a+4:
                    self.all_wall_states.append((a, b))
                    self.wall_act_size += 1

        #벽이 3개 일 때
        for a in range(4, 20):
            if a%4 == 3:
                for b in range(a+1,20):
                    if b%4 == 3:
                        for c in range(b+1,20):
                            self.all_wall_states.append((a, b, c))
                            self.wall_act_size += 1
                    else:
                        for c in range(b+2, 20):
                            self.all_wall_states.append((a, b, c))
                            self.wall_act_size += 1
            else:
                for b in range(a+2, 20):
                    if b%4 == 3:
                        for c in range(b+1,20):
                            self.all_wall_states.append((a, b, c))
                            self.wall_act_size += 1
                    else:
                        for c in range(b+2, 20):
                            self.all_wall_states.append((a, b, c))
                            self.wall_act_size += 1

        for a in range(20, 36):
            for b in range(a+1, 36):
                if b != a+4:
                    for c in range(b+1, 36):
                        if c != a+4 and c != b+4:
                            self.all_wall_states.append((a, b, c))
                            self.wall_act_size += 1
                    
        #벽이 4개 일 때
        for a in range(4, 20):
            if a%4 == 3:
                for b in range(a+1,20):
                    if b%4 == 3:
                        for c in range(b+1,20):
                            if c%4 == 3:
                                for d in range(c+1, 20):
                                    self.all_wall_states.append((a, b, c, d))
                                    self.wall_act_size += 1
                            else:
                                for d in range(c+2, 20):
                                    self.all_wall_states.append((a, b, c, d))
                                    self.wall_act_size += 1
                    else:
                        for c in range(b+2,20):
                            if c%4 == 3:
                                for d in range(c+1, 20):
                                    self.all_wall_states.append((a, b, c, d))
                                    self.wall_act_size += 1
                            else:
                                for d in range(c+2, 20):
                                    self.all_wall_states.append((a, b, c, d))
                                    self.wall_act_size += 1
            else:
                for b in range(a+2, 20):
                    if b%4 == 3:
                        for c in range(b+1,20):
                            if c%4 == 3:
                                for d in range(c+1, 20):
                                    self.all_wall_states.append((a, b, c, d))
                                    self.wall_act_size += 1
                            else:
                                for d in range(c+2, 20):
                                    self.all_wall_states.append((a, b, c, d))
                                    self.wall_act_size += 1
                    else:
                        for c in range(b+2, 20):
                            if c%4 == 3:
                                for d in range(c+1, 20):
                                    self.all_wall_states.append((a, b, c, d))
                                    self.wall_act_size += 1
                            else:
                                for d in range(c+2, 20):
                                    self.all_wall_states.append((a, b, c, d))
                                    self.wall_act_size += 1

        for a in range(20, 36):
            for b in range(a+1, 36):
                if b != a+4:
                    for c in range(b+1, 36):
                        if c != a+4 and c != b+4:
                            for d in range(c+1, 36):
                                if d != a+4 and d != b+4 and d != c+4:
                                    self.all_wall_states.append((a, b, c, d))
                                    self.wall_act_size += 1

        #벽이 5개 일 때
        for a in range(4, 20):
            if a%4 == 3:
                for b in range(a+1,20):
                    if b%4 == 3:
                        for c in range(b+1,20):
                            if c%4 == 3:
                                for d in range(c+1, 20):
                                    if d%4 == 3:
                                        for e in range(d+1, 20):
                                            self.all_wall_states.append((a, b, c, d, e))
                                            self.wall_act_size += 1
                                    else:
                                        for e in range(d+2, 20):
                                            self.all_wall_states.append((a, b, c, d, e))
                                            self.wall_act_size += 1
                            else:
                                for d in range(c+2, 20):
                                    if d%4 == 3:
                                        for e in range(d+1, 20):
                                            self.all_wall_states.append((a, b, c, d, e))
                                            self.wall_act_size += 1
                                    else:
                                        for e in range(d+2, 20):
                                            self.all_wall_states.append((a, b, c, d, e))
                                            self.wall_act_size += 1
                    else:
                        for c in range(b+2, 20):
                            if c%4 == 3:
                                for d in range(c+1, 20):
                                    if d%4 == 3:
                                        for e in range(d+1, 20):
                                            self.all_wall_states.append((a, b, c, d, e))
                                            self.wall_act_size += 1
                                    else:
                                        for e in range(d+2, 20):
                                            self.all_wall_states.append((a, b, c, d, e))
                                            self.wall_act_size += 1
                            else:
                                for d in range(c+2, 20):
                                    if d%4 == 3:
                                        for e in range(d+1, 20):
                                            self.all_wall_states.append((a, b, c, d, e))
                                            self.wall_act_size += 1
                                    else:
                                        for e in range(d+2, 20):
                                            self.all_wall_states.append((a, b, c, d, e))
                                            self.wall_act_size += 1
            else:
                for b in range(a+2, 20):
                    if b%4 == 3:
                        for c in range(b+1,20):
                            if c%4 == 3:
                                for d in range(c+1, 20):
                                    if d%4 == 3:
                                        for e in range(d+1, 20):
                                            self.all_wall_states.append((a, b, c, d, e))
                                            self.wall_act_size += 1
                                    else:
                                        for e in range(d+2, 20):
                                            self.all_wall_states.append((a, b, c, d, e))
                                            self.wall_act_size += 1
                            else:
                                for d in range(c+2, 20):
                                    if d%4 == 3:
                                        for e in range(d+1, 20):
                                            self.all_wall_states.append((a, b, c, d, e))
                                            self.wall_act_size += 1
                                    else:
                                        for e in range(d+2, 20):
                                            self.all_wall_states.append((a, b, c, d, e))
                                            self.wall_act_size += 1
                    else:
                        for c in range(b+2, 20):
                            if c%4 == 3:
                                for d in range(c+1, 20):
                                    if d%4 == 3:
                                        for e in range(d+1, 20):
                                            self.all_wall_states.append((a, b, c, d, e))
                                            self.wall_act_size += 1
                                    else:
                                        for e in range(d+2, 20):
                                            self.all_wall_states.append((a, b, c, d, e))
                                            self.wall_act_size += 1
                            else:
                                for d in range(c+2, 20):
                                    if d%4 == 3:
                                        for e in range(d+1, 20):
                                            self.all_wall_states.append((a, b, c, d, e))
                                            self.wall_act_size += 1
                                    else:
                                        for e in range(d+2, 20):
                                            self.all_wall_states.append((a, b, c, d, e))
                                            self.wall_act_size += 1

        for a in range(20, 36):
            for b in range(a+1, 36):
                if b != a+4:
                    for c in range(b+1, 36):
                        if c != a+4 and c != b+4:
                            for d in range(c+1, 36):
                                if d != a+4 and d != b+4 and d != c+4:
                                    for e in range(d+1, 36):
                                        if e != a+4 and e != b+4 and e != c+4 and e != d+4:
                                            self.all_wall_states.append((a, b, c, d, e))
                                            self.wall_act_size += 1

        #벽이 6개 일 때
        for a in range(4, 20):
            if a%4 == 3:
                for b in range(a+1,20):
                    if b%4 == 3:
                        for c in range(b+1,20):
                            if c%4 == 3:
                                for d in range(c+1, 20):
                                    if d%4 == 3:
                                        for e in range(d+1, 20):
                                            if e%4 == 3:
                                                for f in range(e+1, 20):
                                                    self.all_wall_states.append((a, b, c, d, e, f))
                                                    self.wall_act_size += 1
                                            else:
                                                for f in range(e+2, 20):
                                                    self.all_wall_states.append((a, b, c, d, e, f))
                                                    self.wall_act_size += 1
                                    else:
                                        for e in range(d+2, 20):
                                            if e%4 == 3:
                                                for f in range(e+1, 20):
                                                    self.all_wall_states.append((a, b, c, d, e, f))
                                                    self.wall_act_size += 1
                                            else:
                                                for f in range(e+2, 20):
                                                    self.all_wall_states.append((a, b, c, d, e, f))
                                                    self.wall_act_size += 1
                            else:
                                for d in range(c+2, 20):
                                    if d%4 == 3:
                                        for e in range(d+1, 20):
                                            if e%4 == 3:
                                                for f in range(e+1, 20):
                                                    self.all_wall_states.append((a, b, c, d, e, f))
                                                    self.wall_act_size += 1
                                            else:
                                                for f in range(e+2, 20):
                                                    self.all_wall_states.append((a, b, c, d, e, f))
                                                    self.wall_act_size += 1
                                    else:
                                        for e in range(d+2, 20):
                                            if e%4 == 3:
                                                for f in range(e+1, 20):
                                                    self.all_wall_states.append((a, b, c, d, e, f))
                                                    self.wall_act_size += 1
                                            else:
                                                for f in range(e+2, 20):
                                                    self.all_wall_states.append((a, b, c, d, e, f))
                                                    self.wall_act_size += 1
                    else:
                        for c in range(b+2,20):
                            if c%4 == 3:
                                for d in range(c+1, 20):
                                    if d%4 == 3:
                                        for e in range(d+1, 20):
                                            if e%4 == 3:
                                                for f in range(e+1, 20):
                                                    self.all_wall_states.append((a, b, c, d, e, f))
                                                    self.wall_act_size += 1
                                            else:
                                                for f in range(e+2, 20):
                                                    self.all_wall_states.append((a, b, c, d, e, f))
                                                    self.wall_act_size += 1
                                    else:
                                        for e in range(d+2, 20):
                                            if e%4 == 3:
                                                for f in range(e+1, 20):
                                                    self.all_wall_states.append((a, b, c, d, e, f))
                                                    self.wall_act_size += 1
                                            else:
                                                for f in range(e+2, 20):
                                                    self.all_wall_states.append((a, b, c, d, e, f))
                                                    self.wall_act_size += 1
                            else:
                                for d in range(c+2, 20):
                                    if d%4 == 3:
                                        for e in range(d+1, 20):
                                            if e%4 == 3:
                                                for f in range(e+1, 20):
                                                    self.all_wall_states.append((a, b, c, d, e, f))
                                                    self.wall_act_size += 1
                                            else:
                                                for f in range(e+2, 20):
                                                    self.all_wall_states.append((a, b, c, d, e, f))
                                                    self.wall_act_size += 1
                                    else:
                                        for e in range(d+2, 20):
                                            if e%4 == 3:
                                                for f in range(e+1, 20):
                                                    self.all_wall_states.append((a, b, c, d, e, f))
                                                    self.wall_act_size += 1
                                            else:
                                                for f in range(e+2, 20):
                                                    self.all_wall_states.append((a, b, c, d, e, f))
                                                    self.wall_act_size += 1
            else:
                for b in range(a+2,20):
                    if b%4 == 3:
                        for c in range(b+1,20):
                            if c%4 == 3:
                                for d in range(c+1, 20):
                                    if d%4 == 3:
                                        for e in range(d+1, 20):
                                            if e%4 == 3:
                                                for f in range(e+1, 20):
                                                    self.all_wall_states.append((a, b, c, d, e, f))
                                                    self.wall_act_size += 1
                                            else:
                                                for f in range(e+2, 20):
                                                    self.all_wall_states.append((a, b, c, d, e, f))
                                                    self.wall_act_size += 1
                                    else:
                                        for e in range(d+2, 20):
                                            if e%4 == 3:
                                                for f in range(e+1, 20):
                                                    self.all_wall_states.append((a, b, c, d, e, f))
                                                    self.wall_act_size += 1
                                            else:
                                                for f in range(e+2, 20):
                                                    self.all_wall_states.append((a, b, c, d, e, f))
                                                    self.wall_act_size += 1
                            else:
                                for d in range(c+2, 20):
                                    if d%4 == 3:
                                        for e in range(d+1, 20):
                                            if e%4 == 3:
                                                for f in range(e+1, 20):
                                                    self.all_wall_states.append((a, b, c, d, e, f))
                                                    self.wall_act_size += 1
                                            else:
                                                for f in range(e+2, 20):
                                                    self.all_wall_states.append((a, b, c, d, e, f))
                                                    self.wall_act_size += 1
                                    else:
                                        for e in range(d+2, 20):
                                            if e%4 == 3:
                                                for f in range(e+1, 20):
                                                    self.all_wall_states.append((a, b, c, d, e, f))
                                                    self.wall_act_size += 1
                                            else:
                                                for f in range(e+2, 20):
                                                    self.all_wall_states.append((a, b, c, d, e, f))
                                                    self.wall_act_size += 1
                    else:
                        for c in range(b+2,20):
                            if c%4 == 3:
                                for d in range(c+1, 20):
                                    if d%4 == 3:
                                        for e in range(d+1, 20):
                                            if e%4 == 3:
                                                for f in range(e+1, 20):
                                                    self.all_wall_states.append((a, b, c, d, e, f))
                                                    self.wall_act_size += 1
                                            else:
                                                for f in range(e+2, 20):
                                                    self.all_wall_states.append((a, b, c, d, e, f))
                                                    self.wall_act_size += 1
                                    else:
                                        for e in range(d+2, 20):
                                            if e%4 == 3:
                                                for f in range(e+1, 20):
                                                    self.all_wall_states.append((a, b, c, d, e, f))
                                                    self.wall_act_size += 1
                                            else:
                                                for f in range(e+2, 20):
                                                    self.all_wall_states.append((a, b, c, d, e, f))
                                                    self.wall_act_size += 1
                            else:
                                for d in range(c+2, 20):
                                    if d%4 == 3:
                                        for e in range(d+1, 20):
                                            if e%4 == 3:
                                                for f in range(e+1, 20):
                                                    self.all_wall_states.append((a, b, c, d, e, f))
                                                    self.wall_act_size += 1
                                            else:
                                                for f in range(e+2, 20):
                                                    self.all_wall_states.append((a, b, c, d, e, f))
                                                    self.wall_act_size += 1
                                    else:
                                        for e in range(d+2, 20):
                                            if e%4 == 3:
                                                for f in range(e+1, 20):
                                                    self.all_wall_states.append((a, b, c, d, e, f))
                                                    self.wall_act_size += 1
                                            else:
                                                for f in range(e+2, 20):
                                                    self.all_wall_states.append((a, b, c, d, e, f))
                                                    self.wall_act_size += 1

        for a in range(20, 36):
            for b in range(a+1, 36):
                if b != a+4:
                    for c in range(b+1, 36):
                        if c != a+4 and c != b+4:
                            for d in range(c+1, 36):
                                if d != a+4 and d != b+4 and d != c+4:
                                    for e in range(d+1, 36):
                                        if e != a+4 and e != b+4 and e != c+4 and e != d+4:
                                            for f in range(e+1, 36):
                                                if f != a+4 and f != b+4 and f != c+4 and f != d+4 and f != e+4:
                                                    self.all_wall_states.append((a, b, c, d, e, f))
                                                    self.wall_act_size += 1


        self.q_table_1 = np.zeros((width, width, self.wall_act_size, 4 + (width-1) * (width-1) * 2)) # agent1의 q벨류를 저장하는 변수. 모두 0으로 초기화. 
        self.q_table_2 = np.zeros((width, width, self.wall_act_size, 4 + (width-1) * (width-1) * 2)) # agent2의 q벨류를 저장하는 변수. 모두 0으로 초기화.
        
    def select_action(self, state, agent_num):
        # eps-greedy로 액션을 선택
        coin = random.random()
        if (agent_num != AGENT_1):
            #if coin < self.eps:
            action = random.randint(0,3 + (self.width-1) * (self.width-1) * 2)
            #else:
            #    action_val = self.q_table_2[state[1][1][0],state[1][1][1],:]
            #    action = np.argmax(action_val)
            return action
        else:
            if coin < self.eps:
                action = random.randint(0,3 + (self.width-1) * (self.width-1) * 2)
            else:
                action_val = self.q_table_1[state[1][0][0],state[1][0][1],self.wall_num_prime,:]
                action = np.argmax(action_val)
            return action

    def update_table(self, state, a, r, s_prime, agent_num):
        # 한 에피소드에 해당하는 history를 입력으로 받아 q 테이블의 값을 업데이트 한다
        if (agent_num != AGENT_1):
            for i in range(self.wall_act_size):
                if self.wall_states == self.all_wall_states[i]:
                    self.wall_num = i
        
            if a in range(4, 36):
                for i in range(self.wall_act_size):
                    if self.wall_states == self.all_wall_states[i]:
                        self.wall_num_prime = i
            #QLearning 업데이트 식을 이용
            self.q_table_2[state[1][1][0],state[1][1][1],self.wall_num,a] = self.q_table_2[state[1][1][0],state[1][1][1],self.wall_num,a] + self.alpha * (r + np.amax(self.q_table_2[s_prime[1][1][0],s_prime[1][1][1],self.wall_num_prime,:]) - self.q_table_2[state[1][1][0],state[1][1][1],self.wall_num,a])
            
        else:
            for i in range(self.wall_act_size):
                if self.wall_states == self.all_wall_states[i]:
                    self.wall_num = i
        
            if a in range(4, 36):
                for i in range(self.wall_act_size):
                    if self.wall_states == self.all_wall_states[i]:
                        self.wall_num_prime = i
            #QLearning 업데이트 식을 이용
            self.q_table_1[state[1][0][0],state[1][0][1],self.wall_num,a] = self.q_table_1[state[1][0][0],state[1][0][1],self.wall_num,a] + self.alpha * (r + np.amax(self.q_table_1[s_prime[1][0][0],s_prime[1][0][1],self.wall_num_prime,:]) - self.q_table_1[state[1][0][0],state[1][0][1],self.wall_num,a])
            
        #print(self.q_table_1)

    def anneal_eps(self):
        self.eps -= 0.0001
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
    p1win=0
    p2win=0
    p1rsum=0
    

    for _ in range(1000): # 총 1,000 에피소드 동안 학습
        done = 0
        env.reset()
        agent_1 = env.register_agent() #agent 등록
        agent_2 = env.register_agent()
        walk=0

        while done == 0: # 한 에피소드가 끝날 때 까지
            while True: 
                a_1 = agent.select_action(env.get_state(agent_1), agent_1) #agent1 가능한 action 선택
                
                if a_1 in env.get_legal_action(env.get_state(agent_1)):                
                    break
            
            s_prime_1, r_1, done = env.step(agent_1, a_1) #action 진행 후 state, reward, done 반환
            agent.update_table(env.get_state(agent_1),a_1,r_1,s_prime_1, agent_1)
            walk+=1
            p1rsum+=r_1

            if done == AGENT_1:
                break
            while True: #agent2 가능한 action 선택
                a_2 = agent.select_action(env.get_state(agent_2), agent_2)

                if a_2 in env.get_legal_action(env.get_state(agent_2)):
                    break

            s_prime_2, r_2, done = env.step(agent_2,a_2)
            #agent.update_table(env.get_state(agent_2),a_2,r_2,s_prime_2, agent_2)
            walk+=1
        if done == AGENT_1:
            p1win += 1
            print("episode :", sr, "The number of step : ", walk, "result : p1승리")
        elif done == AGENT_2:
            p2win += 1
            print("episode :", sr, "The number of step : ", walk, "result : p2승리")
        sr+=1

        agent.anneal_eps()
    print("p1승리 : ", p1win, "| p2승리", p2win, "| p1 reward 합 : ",p1rsum)
    env.render(agent_1)
    agent.show_table() # 학습이 끝난 결과를 출력

if __name__ == '__main__':
    main()