import collections
import pygame

MOVE_CNT = 4

DEFAULT_REWARD = -1
PENALTY_REWARD = -100
WIN_REWARD = 100

# 상하좌우
ACTION_OFFSET = [
    [-1, 0],
    [1, 0],
    [0, -1],
    [0, 1]
]


class Quoridor():
    def __init__(self, W, WALL_CNT):
        self.w = W
        self.wall_cnt = WALL_CNT
        self.wall_width_cnt = W - 1

        self.reset()
        self.action_space = list(
            range(0, MOVE_CNT + (self.wall_width_cnt * self.wall_width_cnt) + (self.wall_width_cnt * self.wall_width_cnt)))

    def init_rendering(self):
        pygame.init()
        self.screen = pygame.display.set_mode(
            ((self.w * 40) + ((self.w - 1) * 10), (self.w * 40) + ((self.w - 1) * 10)))
        pygame.display.set_caption("Quoridor")
        self.clock = pygame.time.Clock()

    def reset(self):
        # player [x, y, 남은 벽 설치 개수]
        self.p1_turn = True
        self.player = [[0, self.w//2, self.wall_cnt],
                       [self.w - 1, self.w//2, self.wall_cnt]]
        self.wall_state = [
            [False] * self.wall_width_cnt for _ in range(self.wall_width_cnt * 2)]

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Background color
        self.screen.fill((50, 108, 168))

        # Draw grid
        for i in range(self.w):
            for j in range(self.w):
                pygame.draw.rect(self.screen, (255, 255, 255),
                                 (i * 50, j * 50, 40, 40), border_radius=3)

        # Draw wall
        for i in range(self.wall_width_cnt):
            for j in range(self.wall_width_cnt):
                if self.wall_state[i][j]:
                    pygame.draw.rect(self.screen, (207, 107, 50),
                                     (i * 50 + 40, j * 50, 10, 90))
                if self.wall_state[i + self.wall_width_cnt][j]:
                    pygame.draw.rect(self.screen, (207, 107, 50),
                                     (i * 50, j * 50 + 40, 90, 10))

        pygame.draw.circle(self.screen, (255, 0, 0),
                           (self.player[0][1] * 50 + 20, self.player[0][0] * 50 + 20), 15)
        pygame.draw.circle(self.screen, (0, 0, 255),
                           (self.player[1][1] * 50 + 20, self.player[1][0] * 50 + 20), 15)

        pygame.display.flip()

    def step(self, action):
        done = False
        reward = DEFAULT_REWARD
        if action < MOVE_CNT:
            reward, done = self.move(action)
        elif action < MOVE_CNT + (self.wall_width_cnt * self.wall_width_cnt):
            reward = self.put_wall(action - MOVE_CNT, True)
        elif action < MOVE_CNT + (self.wall_width_cnt * self.wall_width_cnt) + (self.wall_width_cnt * self.wall_width_cnt):
            reward = self.put_wall(
                action - MOVE_CNT - (self.wall_width_cnt * self.wall_width_cnt), False)
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
        new_x = x + ACTION_OFFSET[action][0]
        new_y = y + ACTION_OFFSET[action][1]

        if new_x < 0 or new_x >= self.w or new_y < 0 or new_y >= self.w:
            return PENALTY_REWARD, False

        if action == 0:
            if x != 0:
                if y == 0:
                    if self.wall_state[x + self.wall_width_cnt-1][y]:
                        return PENALTY_REWARD, False
                else:
                    if self.wall_state[x + self.wall_width_cnt-1][y-1] or self.wall_state[x + self.wall_width_cnt-1][y]:
                        return PENALTY_REWARD, False
        elif action == 1:
            if x != self.w - 1:
                if y == 0:
                    if self.wall_state[x + self.wall_width_cnt][y]:
                        return PENALTY_REWARD, False
                else:
                    if self.wall_state[x + self.wall_width_cnt][y-1] or self.wall_state[x + self.wall_width_cnt][y]:
                        return PENALTY_REWARD, False
        elif action == 2:
            if y != 0:
                if x == 0:
                    if self.wall_state[x - 1][y]:
                        return PENALTY_REWARD, False
                else:
                    if self.wall_state[x - 1][y] or self.wall_state[x - 2][y]:
                        return PENALTY_REWARD, False
        elif action == 3:
            if y != self.w - 1:
                if x == 0:
                    if self.wall_state[x][y]:
                        return PENALTY_REWARD, False
                else:
                    if self.wall_state[x][y] or self.wall_state[x - 1][y]:
                        return PENALTY_REWARD, False

        self.player[num][0] = new_x
        self.player[num][1] = new_y

        if self.p1_turn and self.player[num][0] == self.w-1:
            return WIN_REWARD, True
        if not self.p1_turn and self.player[num][0] == 0:
            return WIN_REWARD, True

        return DEFAULT_REWARD, False

    def put_wall(self, action, is_column):
        if self.p1_turn:
            num = 0
        else:
            num = 1

        if self.player[num][2] <= 0:
            return PENALTY_REWARD

        x = action // self.wall_width_cnt
        y = action % self.wall_width_cnt

        print(x, y, is_column)

        if is_column:
            if self.wall_state[x][y]:
                return PENALTY_REWARD
            if x + self.wall_width_cnt == self.wall_width_cnt:
                if self.wall_state[x + self.wall_width_cnt][y]:
                    return PENALTY_REWARD
            else:
                if self.wall_state[x + self.wall_width_cnt][y] or self.wall_state[x + self.wall_width_cnt - 1][y]:
                    return PENALTY_REWARD

            if not self.is_reachable(action, is_column):
                return PENALTY_REWARD

            self.wall_state[x + self.wall_width_cnt][y] = True
        else:
            if self.wall_state[x + self.wall_width_cnt][y]:
                return PENALTY_REWARD
            if y == 0:
                if self.wall_state[x][y]:
                    return PENALTY_REWARD
            else:
                if self.wall_state[x][y] or self.wall_state[x][y - 1]:
                    return PENALTY_REWARD

            if not self.is_reachable(action, is_column):
                return PENALTY_REWARD

            self.wall_state[x][y] = True

        self.player[num][2] -= 1

    def is_reachable(self, action, is_column):
        p_reachable = [False, False]

        x = action // self.wall_width_cnt
        y = action % self.wall_width_cnt

        if is_column:
            if self.wall_state[x][y]:
                return False
            if x + self.wall_width_cnt == self.wall_width_cnt:
                if self.wall_state[x + self.wall_width_cnt][y]:
                    return False
            else:
                if self.wall_state[x + self.wall_width_cnt][y] or self.wall_state[x + self.wall_width_cnt - 1][y]:
                    return False

            self.wall_state[x + self.wall_width_cnt][y] = True
        else:
            if self.wall_state[x + self.wall_width_cnt][y]:
                return False
            if y == 0:
                if self.wall_state[x][y]:
                    return False
            else:
                if self.wall_state[x][y] or self.wall_state[x][y - 1]:
                    return False

            self.wall_state[x][y] = True

        num = 0
        x, y = self.player[num][0], self.player[num][1]

        visited = [[False] * self.w for _ in range(self.w)]
        visited[x][y] = True
        queue = collections.deque()
        queue.append((x, y))

        while queue:
            x, y = queue.popleft()

            if (x == 0 and not num) or (x == self.w - 1 and num):
                p_reachable[num] = True
                break

            for i in range(4):
                new_x = x + ACTION_OFFSET[i][0]
                new_y = y + ACTION_OFFSET[i][1]

                if new_x < 0 or new_x > self.w - 1 or new_y < 0 or new_y > self.w - 1:
                    continue

                if i == 0 and self.wall_state[(x-1) + self.wall_width_cnt][y]:
                    continue
                elif i == 1 and self.wall_state[x + self.wall_width_cnt][y]:
                    continue
                elif i == 2 and self.wall_state[x][y-1]:
                    continue
                elif i == 3 and self.wall_state[x][y]:
                    continue

                if visited[new_x][new_y]:
                    continue

                visited[new_x][new_y] = True
                queue.append((new_x, new_y))

        num = 1
        x, y = self.player[num][0], self.player[num][1]

        visited = [[False] * self.w for _ in range(self.w)]
        visited[x][y] = True
        queue = collections.deque()
        queue.append((x, y))

        while queue:
            x, y = queue.popleft()

            if (x == 0 and not num) or (x == self.w - 1 and num):
                p_reachable[num] = True
                break

            for i in range(MOVE_CNT):
                new_x = x + ACTION_OFFSET[i][0]
                new_y = y + ACTION_OFFSET[i][1]

                if new_x < 0 or new_x > self.wall_width_cnt or new_y < 0 or new_y > self.wall_width_cnt:
                    continue

                if i == 0 and self.wall_state[(x-1) + self.wall_width_cnt][y]:
                    continue
                elif i == 1 and self.wall_state[x + self.wall_width_cnt][y]:
                    continue
                elif i == 2 and self.wall_state[x][y-1]:
                    continue
                elif i == 3 and self.wall_state[x][y]:
                    continue

                if visited[new_x][new_y]:
                    continue

                visited[new_x][new_y] = True
                queue.append((new_x, new_y))

        if is_column:
            x = action // self.wall_width_cnt
            y = action % self.wall_width_cnt

            self.wall_state[x + self.wall_width_cnt][y] = False
        else:
            x = action // self.wall_width_cnt
            y = action % self.wall_width_cnt

            self.wall_state[x][y] = False

        if p_reachable[0] and p_reachable[1]:
            return True
        else:
            return False


if __name__ == '__main__':
    game = Quoridor(5, 10)
    game.init_rendering()

    game.render()

    for _ in range(1):
        game.reset()
        done = False
        while not done:
            game.render()
            action = int(input())
            # action = random.randint(
            #     0, MOVE_CNT + column_WALL_CNT + ROW_WALL_CNT - 1)
            state, r, done = game.step(action)
            game.render()
            # print(state, r, done)
            print(action)
            # input()
