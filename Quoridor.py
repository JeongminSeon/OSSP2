import os
import random

MOVE_CNT = 4
COLUM_WALL_CNT = 64
ROW_WALL_CNT = 64

PENALTY_REWARD = -20
DEFAULT_REWARD = -1
WIN_REWARD = 100

# 상하좌우
ACTION_OFFSET = [
    [-1, 0],
    [1, 0],
    [0, -1],
    [0, 1]
]


class Quoridor():
    def __init__(self):
        self.reset()
        self.action_space = list(
            range(0, MOVE_CNT + COLUM_WALL_CNT + ROW_WALL_CNT))

    def reset(self):
        # player [x, y, 남은 벽 설치 개수]
        self.p1_turn = True
        self.player = [[0, 4, 10], [8, 4, 10]]
        self.wall_state = [[False] * 8 for _ in range(16)]
        self.wall_link = []

    def render(self):
        os.system('cls')
        print(' 0 1 2 3 4 5 6 7 8')
        for i in range(17):
            if i % 2 == 0:
                print(i // 2, end='')
                for j in range(17):
                    if j % 2 == 0:
                        if self.player[0][0] == i//2 and self.player[0][1] == j//2:
                            print("1", end="")
                        elif self.player[1][0] == i//2 and self.player[1][1] == j//2:
                            print("2", end="")
                        else:
                            print("□", end="")
                    else:
                        if i < 2 and self.wall_state[0][j//2]:
                            print("│", end="")
                        elif 2 <= i < 16 and (self.wall_state[i//2][(j-1)//2] or self.wall_state[(i//2)-1][(j-1)//2]):
                            print("│", end="")
                        elif i == 16 and self.wall_state[(i//2)-1][(j-1)//2]:
                            print("│", end="")
                        else:
                            print(" ", end="")
            else:
                print(" ", end="")
                for j in range(17):
                    if j % 2 == 0:
                        if j < 2 and self.wall_state[(i//2) + 8][0]:
                            print("─", end="")
                        elif 2 <= j < 16 and (self.wall_state[(i//2) + 8][j//2] or self.wall_state[(i//2) + 8][(j//2)-1]):
                            print("─", end="")
                        elif j == 16 and self.wall_state[(i//2) + 8][(j//2)-1]:
                            print("─", end="")
                        else:
                            print(" ", end="")
                    else:
                        print(" ", end="")
            print()

    def step(self, action):
        done = False
        reward = DEFAULT_REWARD
        if action < MOVE_CNT:
            reward, done = self.move(action)
        elif action < MOVE_CNT + COLUM_WALL_CNT:
            reward = self.put_wall(action - MOVE_CNT, True)
        elif action < MOVE_CNT + COLUM_WALL_CNT + ROW_WALL_CNT:
            reward = self.put_wall(action - MOVE_CNT - COLUM_WALL_CNT, False)
        else:
            print("error")

        self.p1_turn = not self.p1_turn

        return (self.player[0], self.player[1], self.wall_state), reward, done

    def move(self, action):
        if self.p1_turn:
            num = 0
        else:
            num = 1

        x, y = self.player[num][0], self.player[num][1]
        new_x = self.player[num][0] + ACTION_OFFSET[action][0]
        new_y = self.player[num][1] + ACTION_OFFSET[action][1]

        if new_x < 0 or new_x > 8 or new_y < 0 or new_y > 8:
            return PENALTY_REWARD, False

        if action == 0 and self.wall_state[(x-1)+8][y]:
            return PENALTY_REWARD, False
        elif action == 1 and self.wall_state[x+8][y]:
            return PENALTY_REWARD, False
        elif action == 2 and self.wall_state[x][y-1]:
            return PENALTY_REWARD, False
        elif action == 3 and self.wall_state[x][y]:
            return PENALTY_REWARD, False

        self.player[num][0] = new_x
        self.player[num][1] = new_y

        if self.p1_turn and self.player[num][0] == 8:
            return WIN_REWARD, True
        if not self.p1_turn and self.player[num][0] == 0:
            return WIN_REWARD, True

        return DEFAULT_REWARD, False

    def put_wall(self, action, is_colum):
        if self.p1_turn:
            num = 0
        else:
            num = 1

        if self.player[num][2] <= 0:
            return PENALTY_REWARD

        if is_colum:
            x = action // 8
            y = action % 8

            if self.wall_state[x][y]:
                return PENALTY_REWARD
            if y == 7:
                if self.wall_state[x+8][y]:
                    return PENALTY_REWARD
            else:
                if self.wall_state[x+8][y] or self.wall_state[x+8][y+1]:
                    return PENALTY_REWARD

            self.wall_state[x+8][y] = True
        else:
            x = action // 8
            y = action % 8

            if self.wall_state[x+8][y]:
                return PENALTY_REWARD
            if x == 7:
                if self.wall_state[x][y]:
                    return PENALTY_REWARD
            else:
                if self.wall_state[x][y] or self.wall_state[x+1][y]:
                    return PENALTY_REWARD

            self.wall_state[x][y] = True

        self.player[num][2] -= 1

    def is_reachable(self, action):
        # num: 0 or 1
        # 해당 플레이어가 상대방 진영에 도달할 수 있는지 여부
        print("is_reachable")


if __name__ == '__main__':
    game = Quoridor()
    game.render()

    for _ in range(200):
        game.reset()
        done = False
        while not done:
            game.render()
            action = random.randint(
                0, MOVE_CNT + COLUM_WALL_CNT + ROW_WALL_CNT - 1)
            state, r, done = game.step(action)
            game.render()
            # print(state, r, done)
            print(action)
            input()
