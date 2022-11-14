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
        dist_map = [[-1] * (width) for i in range(width)]
        distance = 0
        next_x = my_pos[0]
        next_y = my_pos[1]
        while True:
            distance += 1
            # NORTH 방향 블럭 거리 갱신
            if (dist_map[my_pos[1] != width-1):  # end_state가 아닐경우 갱신가능
                if (not (my_pos[0] != 0 and board[my_pos[0]-1][my_pos[1]])  # 좌상단에 벽이 있으면 갱신불가
                        and not (my_pos[0] != width-1 and board[my_pos[0]][my_pos[1]])):  # 우상단에 벽이 있으면 갱신 불가
                        TODO

    # 반환값 중복 가능, 자신의 위치 포함 가능
    def reachableAdjacent(self, board, pos_x, pos_y,  opp_pos_x, opp_pos_y):
        width= self.env.w
        ret= []
        if (pos_y < width-1):
            if (not (pos_x != 0 and board[pos_x - 1][pos_y]) and not (pos_x != width-1 and board[pos_x][pos_y])):
                if (opp_pos_x != pos_x and opp_pos_y != pos_y):
                    ret.append([pos_x, pos_y+1])
                elif (pos_y+1 < width-1):  # 진로에 상대 말이 있는 상황
                    if (not (pos_x != 0 and board[pos_x - 1][pos_y+1]) and not (pos_x != width-1 and board[pos_x][pos_y+1])):
                        ret.append([pos_x, pos_y+2])  # 상대말을 건너뛴 위치 도달 가능
                else:  # 상대편 말 건너편을 도달할 수 없는 경우 좌 우로 선회
                    if




        print(dist_map)


def main():
    game= Quoridor(5, 10)
    game.reset()
    agent= DumbAgent(game, 1)
    agent.howFarFromEnd()


if __name__ == '__main__':
    main()
