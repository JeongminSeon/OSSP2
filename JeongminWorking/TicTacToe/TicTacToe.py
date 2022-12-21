
from copy import deepcopy
from mcts import *

# Tic Tac Toe board class
class Board():
    # create constructor (init board class instance)
    def __init__(self, board=None):
        # define players
        self.player_1 = 'x'
        self.player_2 = 'o'
        self.empty_square = '.'

        # define board position
        self.position = {}

        # init (reset) board
        self.init_board()

        # create a copy of a previous board state if available
        if board is not None:
            self.__dict__ = deepcopy(board.__dict__)

    # init (reset) board
    def init_board(self):
        # loop over board rows
        for row in range(3):
            # loop over board columns
            for col in range(3):
                # set every board square to empty square
                self.position[row, col] = self.empty_square

    # make move
    # col : x cordinate, row -> y cordinate
    # this is more human-friendly

    # 사용자 입력을 받은 뒤 x,y -> swap
    def make_move(self, row, col):
        # create new board instance that inherits from the current state
        board = Board(self)

        # make move
        board.position[row, col] = self.player_1

        # swap players
        (board.player_1, board.player_2) = (board.player_2, board.player_1)

        # return new board state
        return board

    def is_draw(self):
       
        for row, col in self.position:
            # empty square is available
            if self.position[row, col] == self.empty_square:
                # this is not a draw
                return False

        return True

    def is_win(self):

        # vertical sequence detection
        for col in range(3):
            # define winning sequence list
            winning_sequence = []

            for row in range(3):
                # if found same next element in a row
                if self.position[row, col] == self.player_2:
                    # update winning sequence
                    winning_sequence.append((row, col))
                # if we have 3 elements in the row
                if len(winning_sequence) == 3:
                    # return the game is won state
                    return True

       
        # horizontal sequence detection
        for row in range(3):
            # define winning sequence list
            winning_sequence = []

            for col in range(3):
                if self.position[row, col] == self.player_2:
                    # update winning sequence
                    winning_sequence.append((row, col))
                # if we have 3 elements in the row
                if len(winning_sequence) == 3:
                    # return the game is won state
                    return True

        # 1st diagnoal sequence detection

        winning_sequence = []

        for row in range(3):
            col = row

            # define winning sequence list
            if self.position[row, col] == self.player_2:
                # update winning sequence
                winning_sequence.append((row, col))
            # if we have 3 elements in the row
            if len(winning_sequence) == 3:
                # return the game is won state
                return True

        # 2nd '' 
        winning_sequence = []

        for row in range(3):
            col = 3 - row - 1

            # define winning sequence list
            if self.position[row, col] == self.player_2:
                # update winning sequence
                winning_sequence.append((row, col))
            # if we have 3 elements in the row
            if len(winning_sequence) == 3:
                # return the game is won state
                return True

        return False

    def generate_states(self):
        # define states list (move list - list of available actions to cosider)
        actions = []

        for row in range(3):
            for col in range(3):
                if self.position[row,col] == self.empty_square:
                    actions.append(self.make_move(row,col))

        # return the list of available actions (board class instances)
        # these actions must be iterable!!!!!
        return actions

    # main game loop
    def game_loop(self):
        print('\n틱택톡 게임 ')
        print(' move를 입력하기 예) 1,1 format [x][y] 종료 : exit ')

        print(self)

        # create MCTS instance
        mcts = MCTS()

        # game loop
        while True:
            # user input
            user_input = input('> ')

            if user_input == 'exit': break
            if user_input == '': continue 
            
            try:
                # parse user input (move format [col, row]: 1,2)
                row = int(user_input.split(',')[1]) - 1
                col = int(user_input.split(',')[0]) - 1

                # check move 
                if self.position[row, col] != self.empty_square:
                    print(' Illegal move!')
                    continue
                
                # make move on baord (사람 입력을 받아서 움직임)
                self = self.make_move(row, col)
                
                # serach for best_move
                best_move = mcts.search(self)

                # legal moves available
                try:
                    # make AI move 
                    self = best_move.board

                # game over
                except:
                    pass

                print(self)

                # check the game state
                if self.is_win():
                    print('player %s 가 게임을 이겼습니다 !!' % self.player_2)
                    break

                elif self.is_draw():
                    print(' Game is drawn! ')
                    break

            except Exception as e:
                print('Error : ', e)
                print('Illegal move')
                print(' Move foramt [x,y] : 1, 2')

    # print board state
    def __str__(self):
        # define board string representation
        board_string = ''

        for row in range(3):
            for col in range(3):
                # 문자열 대입 연산자
                board_string += ' %s' % self.position[row, col]

            board_string += '\n'

        # prepand side to move
        if self.player_1 == 'x':
            board_string = '\n-----------------\n "x" to move:\n-----------------\n\n' + board_string

        elif self.player_1 == 'o':
            board_string = '\n-----------------\n "o" to move:\n-----------------\n\n' + board_string

        return board_string


# main driver
if __name__ == '__main__':
    # create board instance
    board = Board()

    root = TreeNode(board, None)
    root.visits = 6
    root.score = 6

    move_1 = TreeNode(board.generate_states()[0], root)
    move_1.visits = 4
    move_1.score = 4

    move_2 = TreeNode(board.generate_states()[1], root)
    move_2.visits = 2
    move_2.score = 2

    root.children = {
        'child_1' : move_1,
        'child_2' : move_2
    }

    mcts = MCTS()
    # print(board.generate_states()[4])
    best_move = mcts.get_best_move(root, 0)
    
    # print(best_move.score)
    # print(best_move.visits)


    # mcts.rollout(board)
    # print(board)