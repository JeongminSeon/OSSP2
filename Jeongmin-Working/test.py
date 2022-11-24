from QuoridorEnv import *
from mcts_quoridor import *

if __name__ == '__main__':
    # create board instance
    q = QuoridorEnv(width = 5, value_mode= 0)
    agent_1 = q.register_agent()
    # print(agent_1)
    agent_2 = q.register_agent()
    
    mcts = MCTS()

    # mcts.search(q)
    # print(mcts.root.visits)
    mcts.search(q)

    #q.step(agent_1, best_move.get_action())
    #q.render(agent_1)



