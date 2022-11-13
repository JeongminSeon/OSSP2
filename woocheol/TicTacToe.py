class TicTacToe():
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.turn = 1
        self.gameover = False
        return self.board

    def get_state(self):
        if self.turn == 1:
            return self.board
        else:
            new_board = []
            for i in range(9):
                if self.board[i] == 1:
                    new_board.append(2)
                elif self.board[i] == 2:
                    new_board.append(1)
                else:
                    new_board.append(0)
            return new_board

    def step(self, action):
        # 턴 미리 바꿔주기
        self.turn = 3 - self.turn

        # 불가능한 액션
        if self.board[action] != 0:
            return self.get_state(), -10, True
        self.board[action] = self.turn

        # 승리하는 수를 놓았는지
        if self.check_reward():
            return self.get_state(), 2, True

        # 패배하는 수를 놓았는지
        if self.check_defeatable():
            return self.get_state(), -1, False

        # 무승부인지
        if self.check_isfull():
            return self.get_state(), 1, True

        return self.get_state(), 0, False

    def check_isfull(self):
        for i in range(9):
            if self.board[i] == 0:
                return False
        return True

    def check_defeatable(self):
        for i in range(3):
            if (self.board[i * 3] == self.board[i * 3 + 1] == (3 - self.turn) and self.board[i * 3 + 2] == 0) or\
                (self.board[i * 3] == self.board[i * 3 + 2] == (3 - self.turn) and self.board[i * 3 + 1] == 0) or \
                    (self.board[i * 3 + 1] == self.board[i * 3 + 2] == (3 - self.turn) and self.board[i * 3] == 0):
                return True
            if (self.board[i] == self.board[i + 3] == (3 - self.turn) and self.board[i + 6] == 0) or \
                (self.board[i] == self.board[i + 6] == (3 - self.turn) and self.board[i + 3] == 0) or \
                    (self.board[i + 3] == self.board[i + 6] == (3 - self.turn) and self.board[i] == 0):
                return True
        if (self.board[0] == self.board[4] == (3 - self.turn) and self.board[8] == 0) or \
            (self.board[0] == self.board[8] == (3 - self.turn) and self.board[4] == 0) or \
                (self.board[4] == self.board[8] == (3 - self.turn) and self.board[0] == 0):
            return True
        if (self.board[2] == self.board[4] == (3 - self.turn) and self.board[6] == 0) or \
            (self.board[2] == self.board[6] == (3 - self.turn) and self.board[4] == 0) or \
                (self.board[4] == self.board[6] == (3 - self.turn) and self.board[2] == 0):
            return True
        return False

    def available_actions(self):
        actions = []
        for i in range(9):
            if self.board[i] == 0:
                actions.append(i)
        return actions

    def check_reward(self):
        for i in range(3):
            if self.board[i * 3] == self.board[i * 3 + 1] == self.board[i * 3 + 2] != 0:
                return 1
            if self.board[i] == self.board[i + 3] == self.board[i + 6] != 0:
                return 1
        if self.board[0] == self.board[4] == self.board[8] != 0:
            return 1
        if self.board[2] == self.board[4] == self.board[6] != 0:
            return 1
        return 0

    def render(self):
        print(self.turn)
        for i in range(3):
            for j in range(3):
                if self.board[i * 3 + j] == 1:
                    print('O', end='')
                elif self.board[i * 3 + j] == 2:
                    print('X', end='')
                else:
                    print(' ', end='')
            print()
        print()
