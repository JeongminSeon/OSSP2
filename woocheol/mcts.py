
## Monte Carlo tree search (MCTS) 초기 구현


from abc import ABC, abstractmethod
from collections import defaultdict
import math

class MCTS :

    def __init__(self,exploration_weight = 1):
        self.Q = defaultdict(int) # 각 노드의 reward 총합
        self.N = defaultdict(int) # 각 노드의 총 방문횟수
        self.children = dict() # 노드의 자식 노드들
        self.exploration_weight = exploration_weight

    def choose(self, node):
        ## 연속된 노드 중에서 Best인 노드를 선택한다.
        if node.is_terminal():
            # 터미널 노드이면 종료
            raise RuntimeError(f"choose called on terminal node {node}")
        
        if node not in self.children:
            return node.find_random_child()
        
        def score(n):
            if self.N[n] == 0:
                return float("-inf") # 확인 되지 않은 move 포함 X
            return self.Q[n] / self.N[n] # reward의 평균 값

        return max(self.children[node], key = score)

    def do_rollout(self,node):
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node) :
        path = []
        while True :
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node가 탐색되지 않았거나 터미널 노드일 경우
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored :
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node) # depth가 더 높은 layer로 내려간다.

    def _expand(self, node):
        if node in self.children:
            return # 이미 확장된 노드
        self.children[node] = node.find_children()
    
    def _simulate(self,node):
        invert_reward = True
        while True :
            if node.is_termianl():
                reward = node.reward()
                return 1 - reward if invert_reward else reward
            node = node.find_random_child()
            invert_reward = not invert_reward
    
    def _backpropagate(self, path, reward) :
        # leaf 노드의 부모노드로 reward를 전파
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            reward = 1 - reward # 1은 자신 0 은 상대, vice versa

    def _uct_select(self, node):

        # 모든 자식 노드가 이미 확장 되어있아야함.
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            # Upper confidence bound for trees
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )
        
        return max(self.children[node], key = uct)


class Node(ABC):
    ## represent of a single board state
    ## MCTS는 노드들을 확장하면서 작동됨

    @abstractmethod
    def find_children(self):
        ## 이 board state의 모든 가능한 경우
        return set()

    @abstractmethod
    def find_random_child(self):
        ## 효율적인 시뮬레이션을 위해 random child 선택 
        return None

    @abstractmethod
    def is_terminal(self):
        ## 자식 노드가 없으면 터미널
        return True

    @abstractmethod
    def reward(self):
        ## 'self'가 terminal node라고 가정했을 때 : 1=승, 0=패배, .5=무승부, 등등
        return 0

    @abstractmethod
    def __hash__(self):
        ## 노드는 hashable 해야함.
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        ## 노드는 비교 가능해야함.
        return True