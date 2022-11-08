import numpy as np
from Quoridor import Quoridor


class DumbAgent():
    def __init__(self, env, p_num):
        self.p_num = p_num
        self.env = env

    def printMap(self):
        print("1")
        print(self.env.wall_state)
        print(self.env.player)

    def howFarFromEnd(self):
        board = self.env.wall_state
        width = self.env.w
        if self.p_num == 1:
            my_pos = self.env.player[0][0:2]
            opp_pos = self.env.player[1][0:2]
        else:
            my_pos = self.env.player[1][0:2]
            opp_pos = self.env.player[0][0:2]
        dist_map = [[0] * (width) for i in range(width)]
        distance = 0
        while True:
            distance += 1
            # NORTH 방향 블럭 거리 갱신
            if (my_pos[0] != width-1):  # end_state가 아닐경우 갱신가능
                if ((my_pos[0] != 0 and board[my_pos[0]-1][my_pos[1]])  # 좌상단에 벽이 있으면 갱신불가
                        or (my_pos[0] != width-1 and board[my_pos[0]][my_pos[1]])):  # 우상단에 벽이 있으면 갱신 불가

        print(dist_map)


def main():
    game = Quoridor(5, 10)
    game.reset()
    agent = DumbAgent(game, 1)
    agent.howFarFromEnd()


if __name__ == '__main__':
    main()
