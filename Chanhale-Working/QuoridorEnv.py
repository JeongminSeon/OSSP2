import numpy as np

AGENT_1 = 0
AGENT_2 = 1
ACT_MOVE_CNT = 4
ACT_MOVE_NORTH = 0
ACT_MOVE_WEST = 1
ACT_MOVE_SOUTH = 2
ACT_MOVE_EAST = 3

# 약속
# 모든 에이전트는 자신을 p1으로 인식.
# 에이전트의 초기화시 env를 사용하여 자신의 정보를 초기화
# agent 초기화 방법: env.agent1, env.agent2 를 확인.
# 두 값은 boolean 값으로, 이미 차지된 자리는 true 로 표기되어 있음.
# 즉, false 로 표기된 자리가 비어다는 뜻.
# false 로 표기된 값을 true로 세팅하고, 에이전트에 자신이 몇번째 에이전트 자리를 차지했는지 int값으로 보관

# env 를 사용하는 거의 모든 함수의 첫번째 인자는 자신이 몇번째 에이전트인지에 대한 정보가 들어와야 함.
# 아래는 그 목록들임
# step
# get_state

# legal_action
# 아마 많은 에이전트에서 필요로 할 것으로 보여 env 함수에 구현해두었습니다.
# 입력값으로 state를 주면된다. state: (map, player_status)
# state 는 get_state 함수를 이용해 확인하길 바람. numpy array 2개로 이루어진 튜플입니다.


class QuoridorEnv():

    def __init__(self, width=5, value_mode=0):
        if (width > 10 or width < 4 or width % 2 == 0):
            raise Exception(
                '게임판 생성 에러!\n게임판 조건: width < 10 and width > 4 and width % 2 == 1')
        self.width = width
        self.wall_map_width = width - 1
        self.wall_cnt = (width * width) // 8
        self.value_mode = value_mode
        # wall_map_width * wall_map_width * 2 형식의 3차원 배열
        # map: [2][wall_map_width][wall_map_width]이다.
        # [0][][]: 가로로 배치된 벽
        # [1][][]: 새로로 배치된 벽
        self.map = np.array([
            [[False] * self.wall_map_width for _ in range(self.wall_map_width)] for _ in range(2)])
        self.player_status = np.array([[width // 2, 0, self.wall_cnt],
                                       [width // 2, width - 1, self.wall_cnt]])

        # legal 여부와 상관 없이 취할 수 있는 모든 action의 집합 (nwse이동) + (wall 배치 동작 개수)
        self.all_action = np.array(
            [True] * (4 + self.wall_map_width * self.wall_map_width * 2))

    def get_legal_action(self, state=""):
        if (state == ""):
            state = (self.map, self.player_status)
        my_pos = (state[1][0][0], state[1][0][1])
        opp_pos = (state[1][1][0], state[1][1][1])
        width = self.width
        ret = self.all_action.copy()

        # 이동 가능한지 여부 확인
        # N
        if (my_pos[1] >= width - 1):
            ret[ACT_MOVE_NORTH] = False
        elif ((my_pos[0] != 0 and state[0][0][my_pos[0] - 1][my_pos[1]]) or (my_pos[0] < width - 2 and state[0][0][my_pos[0]][my_pos[1]])):
            ret[ACT_MOVE_NORTH] = False
        # W
        if (my_pos[0] == 0):
            ret[ACT_MOVE_WEST] = False
        elif((my_pos[1] != 0 and state[0][1][my_pos[0] - 1][my_pos[1] - 1]) or (my_pos[0] < width - 2 and state[0][1][my_pos[0] - 1][my_pos[1]])):
            ret[ACT_MOVE_WEST] = False
        # S
        if (my_pos[1] == 0):
            ret[ACT_MOVE_SOUTH] = False
        elif ((my_pos[0] != 0 and state[0][0][my_pos[0] - 1][my_pos[1] - 1]) or (my_pos[0] < width - 2 and state[0][0][my_pos[0]][my_pos[1] - 1])):
            ret[ACT_MOVE_SOUTH] = False
        # E
        if (my_pos[0] >= width - 1):
            ret[ACT_MOVE_EAST] = False
        elif((my_pos[1] != 0 and state[0][1][my_pos[0]][my_pos[1] - 1]) or (my_pos[0] < width - 2 and state[0][1][my_pos[0]][my_pos[1]])):
            ret[ACT_MOVE_EAST] = False

        # 벽설치 가능 여부
        # 가로 벽
        for x in range(width - 1):
            for y in range(width - 1):
                # 가로 벽
                if ((x != 0 and state[0][0][x-1][y]) or state[0][0][x][y] or state[0][1][x][y] or (x < (width - 2) and state[0][0][x+1][y])):
                    ret[ACT_MOVE_CNT + y * (width - 1) + x] = False
                # 새로 벽
                if ((y != 0 and state[0][1][x][y-1]) or state[0][1][x][y] or state[0][0][x][y] or (y < (width - 2) and state[0][1][x][y+1])):
                    ret[ACT_MOVE_CNT + (width - 1) *
                        (width - 1) + y * (width - 1) + x] = False

        # 경로 방해여부 검사
        for x in range(width - 1):
            for y in range(width - 1):
                if (ret[ACT_MOVE_CNT + y * (width - 1) + x]):
                    tmp_map = state[0].copy()
                    tmp_map[0][x][y] = True
                    if(self.ask_how_far((tmp_map, state[1])) == -1 or self.ask_how_far_opp((tmp_map, state[1])) == -1):
                        ret[ACT_MOVE_CNT + y * (width - 1) + x] = False
                if (ret[ACT_MOVE_CNT + (width - 1) * (width - 1) + y * (width - 1) + x]):
                    tmp_map = state[0].copy()
                    tmp_map[1][x][y] = True
                    if(self.ask_how_far((tmp_map, state[1])) == -1 or self.ask_how_far_opp((tmp_map, state[1])) == -1):
                        ret[ACT_MOVE_CNT + (width - 1) * (width - 1) +
                            y * (width - 1) + x] = False
        res = []
        for i, j in enumerate(ret):
            if j:
                res.append(i)
        return res

    def render(self, agent_num):
        state = None
        if (agent_num != 1):
            state = self.get_flipped_state()
        else:
            state = (self.map, self.player_status)
        output = []
        output.append("\n\n=========NORTH=========\n")
        # Map 출력
        for i in reversed(range(self.width)):

           # 가로 wall 을 배치하는 파트
            if (i < self.wall_map_width):
                output.append("  ")
                for j in range(self.wall_map_width):
                    if (j != 0 and state[0][0][j - 1][i]):
                        output.append("===")
                    elif (state[0][0][j][i]):
                        output.append("====")
                        continue
                    else:
                        output.append("   ")
                    if (state[0][1][j][i]):
                        output.append("|")
                    else:
                        output.append(" ")
                if (state[0][0][self.wall_map_width - 1][i]):
                    output.append("===")
                if (not state[0][0][self.wall_map_width - 1][i]):
                    output.append("   ")
                output.append("\n")

            output.append(str(i))
            output.append(" ")
            for j in range(self.width):

                # 플레이어 배치하는 파트
                # p1 의 위치는 빨간색으로 표기
                if(state[1][0][0] == j and state[1][0][1] == i):
                    if (agent_num == 1):
                        output.append('\033[42m' + ' 1 ' + '\033[0m')
                    else:
                        output.append('\033[44m' + ' 1 ' + '\033[0m')
                # p2 의 위치는 파란색으로 표기
                elif(state[1][1][0] == j and state[1][1][1] == i):
                    if (agent_num == 1):
                        output.append('\033[44m' + ' 2 ' + '\033[0m')
                    else:
                        output.append('\033[42m' + ' 2 ' + '\033[0m')
                else:
                    output.append('\033[47m' + '   ' + '\033[0m')

                # 새로 wall 을 배치하는 파트
                if j >= self.wall_map_width:
                    continue
                if i < self.wall_map_width:
                    if (state[0][1][j][i]):
                        output.append("|")
                        continue
                if i != 0 and i <= self.wall_map_width:
                    if (state[0][1][j][i - 1]):
                        output.append("|")
                        continue
                output.append(" ")
            output.append("\n")

        # 첫번째 가이드라인 줄 출력
        output.append("  ")
        for i in range(self.width):
            output.append(" ")
            output.append(str(i))
            output.append("  ")
        output.append("\n")
        output.append("=========SOUTH=========")
        print(''.join(output))

    def step(self, agent_num, action):
        state = None
        width = self.width
        if (agent_num != 1):
            state = self.get_flipped_state()
        else:
            state = (self.map, self.player_status)
        # move action
        if (action < 4):
            if (action == ACT_MOVE_NORTH):
                state[1][0][1] += 1
            if (action == ACT_MOVE_WEST):
                state[1][0][0] -= 1
            if (action == ACT_MOVE_SOUTH):
                state[1][0][1] -= 1
            if (action == ACT_MOVE_EAST):
                state[1][0][0] += 1
        elif (action < self.all_action.size):
            action -= ACT_MOVE_CNT
            state[0][action // ((width - 1) * (width - 1))][(action % ((width - 1)
                                                                       * (width - 1))) % (width - 1)][(action % ((width - 1) * (width - 1))) // (width - 1)] = True

        if (agent_num != 1):
            self.map, self.player_status = self.get_flipped_state(state)
        else:
            self.map, self.player_status = state
        step_reward = self.get_value(agent_num)
        step_done = self.ask_end_state((self.map, self.player_status))
        return state, step_reward, step_done

    def step_move(self, agent_num, action):
        if (agent_num != 1):
            state = self.get_flipped_state()
        else:
            state = (self.map, self.player_status)

    def get_state(self, agent_num):
        if (agent_num != 1):
            return self.get_flipped_state()
        else:
            return (self.map, self.player_status)

    def get_flipped_state(self, state=""):
        if state == "":
            flipped_map = np.flip(self.map.copy(), (1, 2))
            flipped_p_status = np.flip(self.player_status.copy(), 0)
            flipped_p_status[0][0] = self.width - 1 - flipped_p_status[0][0]
            flipped_p_status[0][1] = self.width - 1 - flipped_p_status[0][1]
            flipped_p_status[1][0] = self.width - 1 - flipped_p_status[1][0]
            flipped_p_status[1][1] = self.width - 1 - flipped_p_status[1][1]
        else:
            flipped_map = np.flip(state[0].copy(), (1, 2))
            flipped_p_status = np.flip(state[1].copy(), 0)
            flipped_p_status[0][0] = self.width - 1 - flipped_p_status[0][0]
            flipped_p_status[0][1] = self.width - 1 - flipped_p_status[0][1]
            flipped_p_status[1][0] = self.width - 1 - flipped_p_status[1][0]
            flipped_p_status[1][1] = self.width - 1 - flipped_p_status[1][1]

        return (flipped_map, flipped_p_status)

    # state 를 보고 목적지까지 얼마나 멀리 떨어졌는지 판단
    def ask_how_far(self, state):
        width = self.width
        end_line = width - 1
        # 이미 종료상태에 도달
        if (state[1][0][1] == end_line):
            return 0
        reached_map = np.array([[False] * width for _ in range(width)])
        distance = -1
        stack = []
        reached_map[state[1][0][0]][state[1][0][1]] = True
        stack.append((state[1][0][0], state[1][0][1]))
        while len(stack) != 0:
            next_stack = []
            distance += 1
            for pos in stack:
                # N
                if (pos[1] < width - 1):
                    if(not reached_map[pos[0]][pos[1] + 1]):
                        if(not ((pos[0] != 0 and state[0][0][pos[0] - 1][pos[1]]) or (pos[0] < width - 1 and state[0][0][pos[0]][pos[1]]))):
                            reached_map[pos[0]][pos[1] + 1] = True
                            next_stack.append((pos[0], pos[1] + 1))
                # W
                if (pos[0] != 0):
                    if(not reached_map[pos[0] - 1][pos[1]]):
                        if(not((pos[1] != 0 and state[0][1][pos[0] - 1][pos[1] - 1]) or (pos[1] < width - 1 and state[0][1][pos[0] - 1][pos[1]]))):
                            reached_map[pos[0] - 1][pos[1]] = True
                            next_stack.append((pos[0] - 1, pos[1]))
                # S
                if (pos[1] != 0):
                    if(not reached_map[pos[0]][pos[1] - 1]):
                        if(not((pos[0] != 0 and state[0][0][pos[0] - 1][pos[1] - 1]) or (pos[0] < width - 1 and state[0][0][pos[0]][pos[1] - 1]))):
                            reached_map[pos[0]][pos[1] - 1] = True
                            next_stack.append((pos[0], pos[1] - 1))
                # E
                if (pos[0] < width - 1):
                    if(not reached_map[pos[0] + 1][pos[1]]):
                        if(not((pos[1] != 0 and state[0][1][pos[0]][pos[1] - 1]) or (pos[1] < width - 1 and state[0][1][pos[0]][pos[1]]))):
                            reached_map[pos[0] + 1][pos[1]] = True
                            next_stack.append((pos[0] + 1, pos[1]))
            # 종료상태에 도달한 것이 있는지 검사
            for i in next_stack:
                if (i[1] == end_line):
                    return distance+1
            stack = next_stack
        # 종료 상태에 도달할 수 없는 경우
        return (-1)

    def ask_how_far_opp(self, state):
        width = self.width
        end_line = 0
        # 이미 종료상태에 도달
        if (state[1][1][1] == end_line):
            return 0
        reached_map = np.array([[False] * width for _ in range(width)])
        distance = -1
        stack = []
        reached_map[state[1][1][0]][state[1][1][1]] = True
        stack.append((state[1][1][0], state[1][1][1]))
        while len(stack) != 0:
            next_stack = []
            distance += 1
            for pos in stack:
                # N
                if (pos[1] < width - 1):
                    if(not reached_map[pos[0]][pos[1] + 1]):
                        if(not((pos[0] != 0 and pos[1] < width - 1 and state[0][0][pos[0] - 1][pos[1]]) or (pos[0] < width - 1 and state[0][0][pos[0]][pos[1]]))):
                            reached_map[pos[0]][pos[1] + 1] = True
                            next_stack.append((pos[0], pos[1] + 1))
                # W
                if (pos[0] != 0):
                    if(not reached_map[pos[0] - 1][pos[1]]):
                        if(not((pos[1] != 0 and state[0][1][pos[0] - 1][pos[1] - 1]) or (pos[1] < width - 1 and state[0][1][pos[0] - 1][pos[1]]))):
                            reached_map[pos[0] - 1][pos[1]] = True
                            next_stack.append((pos[0] - 1, pos[1]))
                # S
                if (pos[1] != 0):
                    if(not reached_map[pos[0]][pos[1] - 1]):
                        if(not((pos[0] != 0 and state[0][0][pos[0] - 1][pos[1] - 1]) or (pos[0] < width - 1 and state[0][0][pos[0]][pos[1] - 1]))):
                            reached_map[pos[0]][pos[1] - 1] = True
                            next_stack.append((pos[0], pos[1] - 1))
                # E
                if (pos[0] < (width - 1)):
                    if (not reached_map[pos[0] + 1][pos[1]]):
                        if(not((pos[1] != 0 and state[0][1][pos[0]][pos[1] - 1]) or (pos[1] < width - 1 and state[0][1][pos[0]][pos[1]]))):
                            reached_map[pos[0] + 1][pos[1]] = True
                            next_stack.append((pos[0] + 1, pos[1]))
            # 종료상태에 도달한 것이 있는지 검사
            for i in next_stack:
                if (i[1] == end_line):
                    return distance+1
            stack = next_stack
        # 종료 상태에 도달할 수 없는 경우
        return (-1)

    # get_value
    def get_value(self, agent_num):

        # 1.가장 간단한 value_function
        # 승리시 150
        # 패배시 -150
        # 그외 -1
        isItEnd = self.ask_end_state((self.map, self.player_status))
        if (self.value_mode == 0):
            if(isItEnd == 0):
                return -1
            elif (isItEnd == agent_num):
                return 150
            else:
                return -150

        # 2.약간 복잡한 value_function
        # 승리시 100
        # 패배시 -100
        # 그외 y축 값에 따라 차등지급
        # 산식: reward = ($도착 라인과 거리 (벽무시)) * -1
        # ex) 5 x 5 게임판에서 (1, 2): -2, (3, 4): 200, (4, 3): -1
        if(self.value_mode == 1):
            if(isItEnd == 0):
                if(agent_num == 1):
                    return 1 + self.player_status[0][1] - self.width
                else:
                    return -self.player_status[1][1]
            elif (isItEnd == agent_num):
                return 100
            else:
                return -100

        # 3.조금 더 복잡한 value_function
        # 주의점: nq learning 할때 깊이를 충분히 깊게 탐색해야할 것. (아예 end state까지 탐색을 권장)
        #           이 value_function을 도입한 에이전트는 티배깅(인성질)을 할 가능성이 있음...
        # 승리시 1000
        # 패배시 -1000
        # 그 외 "상대와 나의" y축 값에 따라 차등지급
        # 산식 reward = (상대의 end_line 과의 거리 (벽무시)) - (나의 end_line 과의 거리 (벽무시)) * 2 - 1
        if(self.value_mode == 2):
            if(isItEnd == 0):
                if(agent_num == 1):
                    return self.player_status[1][1] - (self.width - 1 - self.player_status[0][1]) * 2 - 1
                else:
                    return self.width - 1 - self.player_status[0][1] - self.player_status[1][1] * 2 - 1
            elif (isItEnd == agent_num):
                return 1000
            else:
                return -1000

        # 4. 많이 복잡한 value_function
        # 주의점: 연산량이 상당하기에 탐색 범위가 넓으면 학습에 상당한 시간이 걸릴 것임.
        # 장점: 직관적으로 고개가 끄덕여지는 reward 를 반환.
        # 승리시 1000
        # 패배시 -1000
        # 그 외  {승리 조건까지 도달하기에 얼마나 남았는지 벽을 포함하여 연산한 값} * -1
        if(self.value_mode == 3):
            if(isItEnd == 0):
                if(agent_num == 1):
                    return -self.ask_how_far((self.map, self.player_status))
                else:
                    return -self.ask_how_far_opp((self.map, self.player_status))
            elif (isItEnd == agent_num):
                return 1000
            else:
                return -1000

        # 5. 아주 많이 복잡한 value_function
        # 주의점: 연산량이 상당하기에 탐색 범위가 넓으면 학습에 상당한 시간이 걸릴 것임.
        # 장점: 직관적으로 고개가 끄덕여지는 reward 를 반환.
        # 승리시 1000
        # 패배시 -1000
        # 그 외 {상대의 도착까지 남은 수} - {승리 조건까지 도달하기에 얼마나 남았는지 벽을 포함하여 연산한 값} * 2 -1
        if(self.value_mode == 4):
            if(isItEnd == 0):
                if(agent_num == 1):
                    return self.ask_how_far_opp((self.map, self.player_status)) - self.ask_how_far((self.map, self.player_status)) * 2 - 1
                else:
                    return self.ask_how_far((self.map, self.player_status)) - self.ask_how_far_opp((self.map, self.player_status)) * 2 - 1
            elif (isItEnd == agent_num):
                return 1000
            else:
                return -1000

    # 종료상태 여부를 확인하는 메서드
    # 1: 입력된 state상에 state[1][0][]의 주인이 승리
    # 2: 입력된 state상에 state[1][1][]의 주인이 승리

    def ask_end_state(self, state):
        if(state[1][0][1] == self.width-1):
            return 1
        elif (state[1][1][1] == 0):
            return 2
        else:
            return 0


q = QuoridorEnv(width=5, value_mode=1)
print(q.step(1, 0))  # agent_1 이 action 10을 수행
q.step(1, 11)
q.step(1, 16)
q.step(2, 0)
print(q.step(2, 3))
q.render(1)
print(q.ask_how_far_opp(q.get_state(1)))
print(q.ask_how_far(q.get_state(1)))
print(q.get_legal_action())
