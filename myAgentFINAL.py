import numpy as np
import matplotlib.pyplot as plt
import random

playerName = "myAgent"
nPercepts = 75  # This is the number of percepts
nActions = 5    # This is the number of actions
generations = 250
trainingAgent = "random"
trainingSchedule = [(trainingAgent, generations)]

class MyCreature:
    def __init__(self):
        self.chromosome = np.random.rand(10)

        self.hunter = 0  # -- move towards enemy
        self.flee = 1  # -- move from enemy

        self.social = 2  # -- move to friends
        self.antisocial = 3  # -- move from friends

        self.indoors = 4  # -- move to wall
        self.outdoors = 5  # -- move from wall

        self.hungry = 6  # -- move towards food
        self.full = 7  # -- move from food

        self.chomp = 8  # -- eat food
        self.exploration = 9  # -- random movement

    def AgentFunction(self, percepts):
        actions = [0, 0, 0, 0, 0]  # left, up, right, down, eat
        creaturePerc = percepts[:, :, 0]  # creatures. 2,2,0 = player pos. x > 0 = friendly, x < 0 = enemy
        foodPerc = percepts[:, :, 1]  # food. 1 = strawberry, 0 = none
        wallPerc = percepts[:, :, 2]  # walls. 1 = wall, 0 = none

        #  exploration, random direction = random * explore gene
        actions[np.random.randint(0,4)] += np.random.rand() * self.chromosome[6]

        #  sense percepts
        for row in range(5):
            for col in range(5):
                creature = creaturePerc[row][col] if creaturePerc[row][col] < 0 else 0  # X > 0 = FRIEND. X < 0 = ENEMY
                if creature < 0:  # ENEMY

                    # print("ENEMY")
                    self.alterActions(row, col, self.hunter, abs(creature), actions)  # HUNTER GENE
                    self.alterActions(row, col, self.flee, abs(creature), actions)  # FLEE GENE

                elif creature > 0:  # FRIEND

                    # print("FRIEND")
                    self.alterActions(row, col, self.social, creature, actions)  # FRIENDLY GENE
                    self.alterActions(row, col, self.antisocial, creature, actions)  # ANTISOCIAL GENE

                wall = wallPerc[row][col]
                if wall == 1:  # WALL

                    # print("WALL")
                    self.alterActions(row, col, self.indoors, wall, actions)  # INDOORS GENE
                    self.alterActions(row, col, self.outdoors, wall, actions)  # OUTDOORS GENE

                food = foodPerc[row][col]
                if food == 1:  # FOOD

                    # print("FOOD")
                    self.alterActions(row, col, self.hungry, food, actions)  # HUNGRY GENE
                    self.alterActions(row, col, self.full, food, actions)  # FULL GENE
        return actions

    def alterActions(self, col, row, type, percep, actions):
        actionVal = self.chromosome[type] * percep
        towards = type % 2 == 0  # if its an even numbered index gene, it moves towards

        if col == 2 and row == 2:  # center
            actions[4] += self.chromosome[self.chomp] * percep  # CHOMP GENE
        else:
            if towards:  # move towards percepts object
                if self.isLeft(col):  # left
                    actions[0] += actionVal
                else:  # right
                    actions[2] += actionVal

                if self.isDown(row):  # down
                    actions[3] += actionVal
                else:  # up
                    actions[1] += actionVal
            else:  # move from percepts object
                if self.isLeft(col):
                    actions[2] += actionVal
                else:
                    actions[0] += actionVal

                if self.isDown(row):
                    actions[1] += actionVal
                else:
                    actions[3] += actionVal

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

def newGeneration(old_population):

    # ---------------------------------------------------------------------------------------------- THE VALUES --------
    mutationRate = 0.05  # 0 to 1, representing a percentage -- Default 0.2-25
    elitismRate = 0.2  # 0 to 1, representing a percentage -- Default 0.4-5
    N = len(old_population)

    # Fitness for all agents
    fitness = np.zeros((N))

    # --------------------------------------------------------------------------------------- FITNESS FUNCTION ---------
    for n, creature in enumerate(old_population):
        # creature.alive (boolean), creature.turn (int), creature.size (int), creature.strawb_eats (int),
        # creature.enemy_eats (int), creature.squares_visited (int), creature.bounces (int))

        fitness[n] += creature.enemy_eats * 8
        fitness[n] += creature.size * 10
        fitness[n] += 30 if creature.alive else 0
        fitness[n] += creature.strawb_eats * 2

    topIndexes = sorted(range(len(fitness)), key=lambda i: fitness[i])[-int(len(fitness) * elitismRate):]

    # ---------------------------------------------------------------------------------- NEW POPULATION
    new_population = list()
    for n in range(N):
        # Create new creature
        new_creature = MyCreature()
        if n < len(topIndexes):  # elitism (keep fittest n of population)
            nextTopFittest = topIndexes.pop(0)
            new_creature = old_population[nextTopFittest]

        else :  # else breeding
            parent1 = tournamentParents(old_population, fitness)  # tournament selection
            parent2 = tournamentParents(old_population, fitness)  # tournament selection

            new_creature.chromosome = crossoverChromosome(parent1.chromosome, parent2.chromosome)  # crossover
            new_creature.chromosome = mutation(mutationRate, new_creature.chromosome)  # mutate

        new_population.append(new_creature)

    # At the end you need to compute average fitness and return it along with your new population
    avg_fitness = np.mean(fitness)

    return new_population, avg_fitness


# chance = number between 0 and 1 (float)
# Based on the set mutation rate, this method mutates a random single gene in the chromosome.
def mutation(chance, chromosome):
    if np.random.rand() < chance:  # mutation
        chromosome[random.randint(0, len(chromosome)-1)] = np.random.rand()
    return chromosome


# randomly crosses gene pairs,
def crossoverChromosome(chromosome1, chromosome2):
    newChromosome = []
    parents = [chromosome1, chromosome2]
    # crosses entire genes (keeps the pair for/against genes)
    for i in range(0, len(chromosome1), 2):
        parent = random.choice(parents)
        newChromosome.append(parent[i])
        newChromosome.append(parent[i+1])
    return newChromosome


# one parent selected over another based on fitness
def tournamentParents(population, fitness):
    parent1 = population[random.randint(0, len(population)-1)]
    parent2 = population[random.randint(0, len(population)-1)]
    if fitness[population.index(parent1)] > fitness[population.index(parent2)]:
        return parent1
    else:
        return parent2
