import numpy as np
import random

playerName = "myAgentV1"
nPercepts = 75  # This is the number of percepts
nActions = 5    # This is the number of actions

# Train against random for 5 generations, then against self for 1 generations
trainingSchedule = [("random", 5)]
# trainingSchedule = [("random", 10), ("self", 1)]

# This is the class for your creature/agent
# chromosome index 0 = aggression gene (chance of hunting vs running)
# chromosome index 1 = friendly gene (chance of moving towards friends)
# chromosome index 2 = hungry gene (chance of seeking food vs running)
# chromosome index 3 = he HATES walls (chance of moving away from walls)
# chromosome index 4 = eating gene (chance of eating vs ignoring)

class MyCreature:

    def __init__(self):
        # You should initialise self.chromosome member variable here (whatever you choose it
        # to be - a list/vector/matrix of numbers - and initialise it with some random values
        self.chromosome = np.random.rand(5)
        self.chromosome[4] = 100


    def isLeft(self, col):
        if col < 3:  # less than the center column = True (left)
            return True
        else:  # otherwise it must be False (right)
            return False

    def isDown(self, row):
        if row < 3:  # less than the center row = True (down)
            return True
        else:  # otherwise it must be False (up)
            return False

    def modifyActions(self, row, col, percep, actions, type):
        if True:
            if type != 0:
                if row == 3 and col == 3:  # if directly on
                    actions[4] += percep * self.chromosome[4]

                if self.isLeft(col):  # if left
                    actions[0] += percep * self.chromosome[type]
                elif not self.isLeft(col):  # if right
                    actions[2] += percep * self.chromosome[type]
                elif self.isDown(row):  # if down
                    actions[3] += percep * self.chromosome[type]
                elif not self.isDown(row):  # if up
                    actions[1] += percep * self.chromosome[type]
            else:
                if row == 3 and col == 3:  # if directly on
                    actions[4] += percep * self.chromosome[2]

                if self.isLeft(col):  # if left
                    actions[0] -= percep * self.chromosome[type]
                elif not self.isLeft(col):  # if right
                    actions[2] -= percep * self.chromosome[type]
                elif self.isDown(row):  # if down
                    actions[3] -= percep * self.chromosome[type]
                elif not self.isDown(row):  # if up
                    actions[1] -= percep * self.chromosome[type]

        else:  # some random movement function idk
            actions[np.random.randint(0,5)] += np.random.rand()

    def AgentFunction(self, percepts):
        actions = [0, 0, 0, 0, 0]  # left, up, right, down, eat
        creaturePerc = percepts[:, :, 0]  # creatures. 2,2,0 = player pos. x > 0 = friendly, x < 0 = enemy
        foodPerc = percepts[:, :, 1]  # food. 1 = strawberry, 0 = none
        wallPerc = percepts[:, :, 2]  # walls. 1 = wall, 0 = clear

        #  updating variables with current world data
        for row in range(5):
            for col in range(5):
                enemy = creaturePerc[row][col] if creaturePerc[row][col] < 0 else 0  # check for enemy
                if enemy < 0:  # enemy present
                    self.modifyActions(row, col, enemy, actions, 0)

                friend = creaturePerc[row][col] if creaturePerc[row][col] > 0 else 0  # check for friendly
                if friend > 0:
                    self.modifyActions(row, col, friend, actions, 1)

                food = foodPerc[row][col]
                if food > 0:
                    self.modifyActions(row, col, food, actions, 2)

                wall = wallPerc[row][col]
                if wall > 0:
                    self.modifyActions(row, col, food, actions, 3)

        # You should implement a model here that translates from 'percepts' to 'actions'
        # through 'self.chromosome'.

        # Different 'percepts' values should lead to different 'actions'.  This way the agent
        # reacts differently to different situations.

        # Different 'self.chromosome' should lead to different 'actions'.  This way different
        # agents can exhibit different behaviour.

        return actions


def newGeneration(old_population):

    # This function should return a list of 'new_agents' that is of the same length as the list of 'old_agents'.
    # That is, if previous game was played with N agents, the next game should be played with N agents again.

    N = len(old_population)  # This function should also return average fitness of the old_population

    fitness = np.zeros((N))  # Fitness for all agents

    # This loop iterates over your agents in the old population - the purpose of this boiler plate code is
    # to demonstrate how to fetch information from the old_population in order to score fitness of each agent
    for n, creature in enumerate(old_population):
        # creature is an instance of MyCreature that you implemented above, therefore you can access any attributes
        # (such as `self.chromosome').  Additionally, the objects has attributes provided by the game engine:

        # creature.alive (boolean), creature.turn (int), creature.size (int), creature.strawb_eats (int),
        # creature.enemy_eats (int), creature.squares_visited (int), creature.bounces (int))

        # fitness[n] += 80 if creature.alive else (creature.turn * 0.75)
        # fitness[n] += creature.strawb_eats * 50
        # fitness[n] -= creature.bounces
        fitness[n] += creature.enemy_eats + creature.strawb_eats + (creature.turn * 0.25)
        # e.g. creature who survived 74 turns, ate 6 (30) strawberries, ate 7 (70) enemies = 174 points
        # e.g. creature who survived until end (140), ate 2 (10) strawberries, ate 0 enemies = 150 points

    # top 25% indexes
    topIndexes = sorted(range(len(fitness)), key=lambda i: fitness[i])[-int(len(fitness)*0.25):]
    topLen = len(topIndexes)

    new_population = list()  # assignment comment - sort the agent according to fitness and create new population

    for n in range(N):
        new_creature = MyCreature()  # Create new creature

        if n < topLen:  # keep the fittest 25% of population (elitism)
            nextTopFittest = topIndexes.pop(0)
            new_creature = old_population[nextTopFittest]

        else :  # else select other 75% by comparing
            parent1 = old_population[random.randrange(0, len(old_population))]
            parent2 = old_population[random.randrange(0, len(old_population))]

            if fitness[old_population.index(parent1)] > fitness[old_population.index(parent2)]:
                new_creature = parent1
            else:
                new_creature = parent2
            if np.random.rand() > np.random.rand():  # mutation
                new_creature.chromosome[random.randint(0, len(new_creature.chromosome)-1)] = np.random.rand()

        # Here you should modify the new_creature's chromosome by selecting two parents (based on their
        # fitness) and crossing their chromosome to overwrite new_creature.chromosome

        # Consider implementing elitism, mutation and various other strategies for producing new creature.

        new_population.append(new_creature)  # Add the new agent to the new population

    # At the end you need to compute average fitness and return it along with your new population
    avg_fitness = np.mean(fitness)

    return (new_population, avg_fitness)
