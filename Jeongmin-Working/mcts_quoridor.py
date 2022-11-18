
# packages
import math
import random 
from copy import deepcopy
from QuoridorEnv import *

MCTS_ITERATIONS = 500
EXPLORATION_CONSTANT = 2
AGENT_1 = 100
AGENT_2 = 200

# tree node class definition
class TreeNode():
    # class constructor (create tree node class instance)

    def __init__(self, env, parent, action = -1):
        # init Env state
        self.env = env

        # AttributeError: 'TreeNode' object has no attribute 'step_done'
        step_done = env.ask_end_state((env.map, env.player_status))
        # init is node terminal (flag)
        if step_done == AGENT_1 or step_done == AGENT_2:
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

        # parent_action
        self.action = action
    
    def get_action(self):

        return self.action

# MCTS class definition
class MCTS() :
    # search for the best move in the current position
    def search(self, initial_state):
        # create root node
        self.root = TreeNode(initial_state, None)

        for iterations in range(MCTS_ITERATIONS):
            # select a node (selection)
            node = self.select(self.root)
            
            # score current node (simiulation)
            score = self.rollout(node.env)

            # backpropagate results
            self.backpropagate(node, score)

        # pick up the best move in the current position
        try:
            return self.get_best_move(self.root, EXPLORATION_CONSTANT)
        
        except:
            pass
    
    # select most promising node
    def select(self, node):
        # make sure that the node is not terminal state
        while not node.is_terminal:
            # fully exapnded
            # best_move 함수를 통해 가장 유망한 자식 노드를 받을 수 있음
            if node.is_fully_expanded:
                node = self.get_best_move(node, EXPLORATION_CONSTANT)

            # not fully expanded
            else:
                # 완전 확장 될때에만 계산 가능 -> 노드 확장
                return self.expand(node)
        
        # fully exapnded -> 자식 노드 반환
        # not -> 새로 확장된 노드 반환
        return node

    def expand(self, node):
        # generate legal actions (states)
        
        # 현재 player를 기준으로 action 찾기
        current_player = AGENT_1

        if(node.env.last_played == AGENT_1):
            current_player = AGENT_2
        else:
            current_player = AGENT_1
            
        actions = node.env.get_legal_action(node.env.get_state(current_player))
        
        # 액션을 취하고 난 뒤에 env를 저장하기 위한 변수
        node_action =  deepcopy(node)

        # loop over generated actions (states)
        for action in actions :

            # 현재 상태(actions)가 자식노드에 존재하면 안됨
            if action not in node.children:
                
                # 생성된 노드마다 각각의 state(map)을 가지고 있어야하며 달라야함.
                # last_played ? current_player? 이 부분 다시 설정 필요
                # create a new node
                node_action.env.step(node.env.last_played,action)

                new_node = TreeNode(node_action.env, node, action)
                # new_node = TreeNode(node.env, node, action)

                # 자식 노드를 부모 노드 children dict(list) 에 추가
                node.children[action] = new_node

                # no more legal action 
                if len(actions) == len(node.children) :
                    node.is_fully_expanded = True
                
                # new_node.env.render(current_player)
                # 새로 만든 노드 반환
                return new_node

        # debugging
        print('여기 오면 이상한 것')

    # simulate the game via making random actions until reach end of the game
    def rollout(self, env) : 

        if(env.last_played == AGENT_1):
            current_player = AGENT_2
        else:
            current_player = AGENT_1

        # terminal state에 도달할 때까지 랜덤 액션(move)
        while not env.ask_end_state((env.map, env.player_status)):
            try:
                action = random.choice(env.get_legal_actions(env.get_state(current_player)))
                env.step(env.last_played,action)
            # no moves available
            except:
                return 0

        if current_player == AGENT_1 : return 1
        elif current_player == AGENT_2 : return -1


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
    def get_best_move(self, node, c_param = EXPLORATION_CONSTANT):
        # define best score & best moves
        # 음의 무한대 반환
        best_score = float('-inf')
        best_moves = []
        
        # loop over child nodes
        for child_node in node.children.values():
            # define current player
            if child_node.env.last_played == AGENT_1 : current_player = 1
            elif child_node.env.last_played == AGENT_2 : current_player = -1
            
            try:
                # get move score using UCT formula  
                move_score = current_player * child_node.score / child_node.visits + c_param * math.sqrt(math.log(node.visits / child_node.visits))                                        
            except ZeroDivisionError:
                #equivalent to infinity
                move_score = 10000
                                                  
            
            # better move has been found
            if move_score > best_score:
                best_score = move_score
                best_moves = [child_node]
            
            # found as good move as already available
            elif move_score == best_score:
                best_moves.append(child_node)
            
        # return one of the best moves randomly
        return random.choice(best_moves)
