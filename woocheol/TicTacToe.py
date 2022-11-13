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

    # return state, reward, done
    def step(self, action):
        if self.board[action] != 0:
            return self.get_state(), -1, False
        self.board[action] = self.turn
        self.turn = 3 - self.turn
        self.reward = self.check_reward()
        if self.reward or self.check_full():
            self.reward = 1
            self.gameover = True
        return self.get_state(), self.reward, self.gameover

    def check_full(self):
        for i in range(9):
            if self.board[i] == 0:
                return False
        return True

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
