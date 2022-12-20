from QuoridorEnv import *
from mcts_quoridor import *
from MctsAgent import *
from DumbAgent import DumbAgent

if __name__ == '__main__':
    # create board instance
    q = QuoridorEnv(width = 5, value_mode= 0)
    agent_1 = q.register_agent()
    # print(agent_1)
    agent_2 = q.register_agent()

    sr = 1
    p1win = 0
    p2win = 0

    mcts = MCTS()

    for _ in range(10) :
        q.reset()

        while True: # 한 에피소드가 끝날 때 까지
            if q.ask_end_state((q.map, q.player_status)) == AGENT_1:
                p1win += 1
                print("episode :", sr, "result : dumb agent 승리")
                break
            if q.ask_end_state((q.map, q.player_status)) == AGENT_2:
                p2win += 1
                print("episode :", sr, "result : mcts agent 승리")
                break
            
            dumb = DumbAgent(q, agent_1)
            
            action = dumb.get_action()
            
            q.step(agent_1, action)
            q.render(agent_1)

            
            if q.ask_end_state((q.map, q.player_status)) == AGENT_1:
                p1win += 1
                print("episode :", sr, "result : dumb agent 승리")
                break
            if q.ask_end_state((q.map, q.player_status)) == AGENT_2:
                p2win += 1
                print("episode :", sr, "result : mcts agent 승리")
                break

            
            node = mcts.search(q)

            action = node.get_action()
            q.step(agent_2, action)
            
            q.render(agent_1)
        
        sr+=1

    print("p1승리 : ", p1win, "| p2승리 : ", p2win)
