
# packages
import math
import random 

# tree node class definition
class TreeNode():
    # class constructor (create tree node class instance)

    def __init__(self, board, parent):
        # init board state
        self.board = board

        # init is node terminal (flag)
        if self.board.is_win() or self.board.is_draw():
            # we have a terminal node
            self.is_terminal = True
        
        else:
            self.is_terminal = False
        
        # set is fully expanded flag
        self.is_fully_expanded = self.is_terminal
 
        # init parent node if available
        self.parent = parent
        # init the number of node visits
        self.visits = 0

        # init the total score of the node
        self.score = 0

        # init current node's children
        self.children = {}

# MCTS class definition
class MCTS() :
    # search for the best move in the current position
    def search(self, initial_state):
        # create root node
        self.root = TreeNode(initial_state, None)

        for _ in range(1000):
            # select a node (selection)
            node = self.select(self.root)
            
            # score current node (simiulation)
            score = self.rollout(node.board)

            # backpropagate results
            self.backpropagate(node, score)

        # pick up the best move in the current position
        try:
            return self.get_best_move(self.root, 0)
        
        except:
            pass
    
    # select most promising node
    def select(self, node):
        # make sure that the node is not terminal state
        while not node.is_terminal:
            # fully exapnded
            # best_move 함수를 통해 가장 유망한 자식 노드를 받을 수 있음
            if node.is_fully_expanded:
                node = self.get_best_move(node, 2)

            # not fully expanded
            else:
                # 완전 확장 될때에만 계산 가능 -> 노드 확장
                return self.expand(node)
        
        # fully exapnded -> 자식 노드 반환
        # not -> 새로 확장된 노드 반환
        return node

    def expand(self, node):
        # generate legal states (moves)
        states = node.board.generate_states()


        # loop over generated states (moves)
        for state in states :
            print(state)
            # 현재 상태(move)가 자식노드에 존재하면 안됨
            # stringfy해서 dic에서 key로 사용 -> 중복 X
            if str(state.position) not in node.children:
                # create a new node
                new_node = TreeNode(state, node)

                # 자식 노드를 부모 노드 children dict(list) 에 추가
                node.children[str(state.position)] = new_node

                # no more legal action 
                if len(states) == len(node.children) :
                    node.is_fully_expanded = True
                
                print(new_node.board)
                # 새로 만든 노드 반환
                return new_node

        # debugging
        print('여기 오면 이상한 것')

    # simulate the game via making random moves until reach end of the game
    def rollout(self, board) : 
        # terminal state에 도달할 때까지 랜덤 액션(move)
        while not board.is_win():
            try:
                board = random.choice(board.generate_states())

            # no moves available
            except:
                return 0

        if board.player_2 == 'x' : return 1
        elif board.player_2 == 'o' : return -1


    # backpropagate the number of visits and score up to the root node
    def backpropagate(self, node, score):
        while node is not None:
            # update node's visits
            node.visits += 1

            # update node's score
            node.score += score

            # set node to parent
            # 노드 한단계 업 -> 부모 노드까지 올라감.
            node = node.parent


    # select the best node basing on UCB1 formula
    def get_best_move(self, node, exploration_constant):
        # define best score & best moves
        # 음의 무한대 반환
        best_score = float('-inf')
        best_moves = []
        
        # loop over child nodes
        for child_node in node.children.values():
            # define current player
            if child_node.board.player_2 == 'x': current_player = 1
            elif child_node.board.player_2 == 'o': current_player = -1
            
            # get move score using UCT formula
            move_score = current_player * child_node.score / child_node.visits + exploration_constant * math.sqrt(math.log(node.visits / child_node.visits))                                        
            
            # better move has been found
            if move_score > best_score:
                best_score = move_score
                best_moves = [child_node]
            
            # found as good move as already available
            elif move_score == best_score:
                best_moves.append(child_node)
            
        # return one of the best moves randomly
        return random.choice(best_moves)
        