# Game theory
# based on material from https://ncase.me/trust/

from enum import Enum

class Action(Enum):
    cooperate = 0
    cheat     = 1

class ActionCosts:
    def __init__(self, cheat_cost, cooperate_cost, others_cheat_gain, others_coopertate_gain) -> None:
        #how much does each choice cost
        self.cheat_cost = cheat_cost
        self.cooperate_cost = cooperate_cost

        #the other player cheats/cooperates, how much do you get 
        self.others_cheat_gain = others_cheat_gain
        self.others_coopertate_gain = others_coopertate_gain

class Game:
    def __init__(self, matrix) -> None: #supply outcome matrix x2
        #not symmetric, not additive
        self.matrix = matrix

    @classmethod
    def init_alt(cls, outcomes_A, outcomes_B):
        outcomes_B_flip = cls.flip_outcomes(outcomes_B)
        outcomes_combined = [ # zip 2D lists manually
            [(outcomes_A[0][0], outcomes_B_flip[0][0]), (outcomes_A[0][1], outcomes_B_flip[0][1])],
            [(outcomes_A[1][0], outcomes_B_flip[1][0]), (outcomes_A[1][1], outcomes_B_flip[1][1])]
        ]
        return cls(outcomes_combined)

    @classmethod
    def make_symmetric_game(cls, outcomes): #supply outcome matrix x1
        return cls.init_alt(outcomes, outcomes)

    #'additive' outcome matrix can be written as
    # [
    #   [a + c, a + d],
    #   [b + c, b + d]
    # ]
    #  - this has 3 degrees of freedom (a is not needed)
    # otherwise, its not, and needs 4 degrees of freedom
    def cost_to_outcome(costs: ActionCosts):
        outcomes = [
            [costs.others_coopertate_gain - costs.cooperate_cost, costs.others_cheat_gain - costs.cooperate_cost],
            [costs.others_coopertate_gain - costs.cheat_cost,     costs.others_cheat_gain - costs.cheat_cost]
        ]
        return outcomes

    def flip_outcomes(outcomes): #make adjoint of outcome matrix 
        return [
            [outcomes[0][0], outcomes[1][0]],
            [outcomes[0][1], outcomes[1][1]]
        ]
        
    @classmethod
    def make_additive_game(cls, costs_A, costs_B): #supply costs for actions x2
        return cls.init_alt(cls.cost_to_outcome(costs_A), cls.cost_to_outcome(costs_B))

    @classmethod
    def make_symmetric_additive_game(cls, costs): #supply costs for actions x1
        return cls.make_symmetric_game(cls.cost_to_outcome(costs))

# points_for_A, points_for_B = outcome_matrix[action_A][action_B]
#         B coop, B cheat
# A coop  (A, B)
# A cheat


'''
gouda = Cheese()
emmentaler = Cheese.random()
leerdammer = Cheese.slightly_holey()
'''

def test_constructor():
    game1 = Game(
        [[( 2,  2), (-1, 3)],
         [( 3, -1), ( 0, 0)]]
    )

    game2 = Game.make_symmetric_game(
        [[ 2, -1],
         [ 3,  0]]
    )

    game3 = Game.make_symmetric_additive_game(
        ActionCosts(
            cheat_cost = 0,
            cooperate_cost = 1,

            others_cheat_gain = 0,
            others_coopertate_gain = 3
        )
    )

    #print(game1.matrix)
    #print(game2.matrix)
    #print(game3.matrix)

    assert(game1.matrix == game2.matrix)
    assert(game1.matrix == game3.matrix)

test_constructor()


#match between two agents
def match(n_rounds, game: Game, agent_A, agent_B):
    history_A = []
    history_B = []

    total_A = 0
    total_B = 0
    for _ in range(n_rounds):
        action_A = agent_A.action(history_B, history_A)
        action_B = agent_B.action(history_A, history_B)

        history_A += [action_A]
        history_B += [action_B]

        points_A, points_B = game.matrix[action_A.value][action_B.value]

        total_A += points_A
        total_B += points_B

    return (total_A, total_B)

import itertools

def match_all(n_rounds, game: Game, agents):
    combinations = list(itertools.combinations(range(len(agents)), 2))
    totals = [0] * len(agents)
    for agent_A_i, agent_B_i in combinations:
        points_A, points_B = match(n_rounds, game, agents[agent_A_i], agents[agent_B_i])
        totals[agent_A_i] += points_A
        totals[agent_B_i] += points_B

    return totals

from collections import Counter

def evolve(n_rounds, game: Game, agents):
    previous_counter = None
    print("start")
    print(Counter([i.__class__.__name__ for i in agents]))
    datapoints = {k:[v] for k, v in Counter([i.__class__.__name__ for i in agents]).items()}
    round_n = 1
    while(True):
        print(f"round {round_n}")
        points = match_all(n_rounds, game, agents)
        sorted_agents = [x[0] for x in sorted(zip(agents, points), key = lambda x: x[1])]
        print(sorted(zip([i.__class__.__name__ for i in agents], points), key = lambda x: x[1]))
        agents = sorted_agents[5:] + sorted_agents[-5:] # remove lowest performing 5 and duplicate highest performing 5
        
        print(Counter([i.__class__.__name__ for i in agents]))

        counter = Counter([i.__class__.__name__ for i in agents])
        for k in datapoints:
            datapoints[k] += [counter[k] if(k in counter) else 0]

        print(datapoints)
        input()

        if(len(counter) == 1):
            print("game dominated by agent", counter)
            break
        if(previous_counter == counter):
            print("no change since last cycle")
            break
        round_n += 1

        previous_counter = Counter(counter)

'''
class Agent:
    def __init__(self, strategy_function) -> None:
        self.opponent_history = []
        self.own_history = []
        self.strategy_function = strategy_function

    def action(self):
        action = self.strategy_function(self.opponent_history, self.own_history)
        self.own_history += [action]
        return action

    def recieve_responce(self, responce_action):
        self.opponent_history += [responce_action]
'''


game = Game.make_symmetric_additive_game(
    ActionCosts(
        cheat_cost = 0,
        cooperate_cost = 1,

        others_cheat_gain = 0,
        others_coopertate_gain = 3
    )
)

class Always_cheat:
    def action(self, x, y):
        return Action.cheat

class Always_cooperate:
    def action(self, x, y):
        return Action.cooperate

from random import random

class Random_outcome:
    def __init__(self, chance_to_cheat) -> None:
        self.chance_to_cheat = chance_to_cheat
        
    def action(self, x, y):
        return [Action.cooperate, Action.cheat][random() < self.chance_to_cheat]

class Copycat:
    def action(self, opponent_history, own_history):
        return ([Action.cooperate] + opponent_history)[-1]

class Grudger:
    def action(self, opponent_history, own_history):
        if Action.cheat in opponent_history:
            return Action.cheat
        else:
            return Action.cooperate

agents = [Always_cheat() for _ in range(4)] + \
         [Always_cooperate() for _ in range(12)] + \
         [Copycat() for _ in range(4)] + \
         [Grudger() for _ in range(6)] + \
         [Random_outcome(0.5) for _ in range(6)]

evolve(10, game, agents)

