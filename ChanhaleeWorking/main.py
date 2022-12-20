from HeuristicAgent import HeuristicAgent
from Quoridor import Quoridor


def main():
    game = Quoridor(5, 10)
    game.reset()
    agent = HeuristicAgent(game, 1)


if __name__ == '__main__':
    main()
