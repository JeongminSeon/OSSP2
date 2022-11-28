from QuoridorEnv import *
from mcts_quoridor import *

if __name__ == '__main__':
    # create board instance
    q = QuoridorEnv(width = 5, value_mode= 0)
    agent_1 = q.register_agent()
    # print(agent_1)
    agent_2 = q.register_agent()
    
    mcts = MCTS()
    
    q.step(agent_1, 20)
    q.step(agent_1, 27)
    q.step(agent_1, 28)
    q.last_played = agent_2

    test_node = TreeNode(q, None)
    test_node.env.render(agent_1)

    node = mcts.search(q)
    
    node.env.render(agent_1)
    
    # q.step(agent_1, 0)
    # Test_node = TreeNode(q, root, 0)

    # best_node = mcts.search(q)

    
    # root = TreeNode(q, None)
    
    # expand Test O
    # for _ in range(0,50) :
    #     if root.is_fully_expanded == True :
    #         pass
    #     else :
    #         node = mcts.expand(root)
    #         node.env.render(agent_1)
    
    # q.step(agent_1, 0)
    # Test_node = TreeNode(q, root, 0)
    # # rollout Test

    # count = 1
    # terminal state에 도달할 때까지 랜덤 액션(move)
    # while Test_node.env.ask_end_state((Test_node.env.map, Test_node.env.player_status)) == False:
    #     try:
    #         if(Test_node.env.last_played == AGENT_1):
    #             current_player = AGENT_2
    #         else:
    #             current_player = AGENT_1

    #         actions = Test_node.env.get_legal_action(Test_node.env.get_state(current_player))
    #         action = random.choice(actions)
    #         Test_node.env.step(current_player,action)
    #         Test_node.env.render(current_player)
    #     # no moves available
    #     except:
    #         pass 

    # # last_played 조정 필요(아직 불확실)
    # # terminal state에 도달 승리시 1, 패배시 -1
            
    # if Test_node.env.last_played == AGENT_1 : score =  1
    # elif Test_node.env.last_played == AGENT_2 : score = -1
    # print(score)
    #q.step(agent_1, best_move.get_action())
    #q.render(agent_1)



