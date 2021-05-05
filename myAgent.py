import numpy as np
import random

playerName = "myAgent"
nPercepts = 75  # This is the number of percepts
nActions = 5    # This is the number of actions

trainingSchedule = [("random", 51)]

class MyCreature:

    def __init__(self):
        # You should initialise self.chromosome member variable here (whatever you choose it
        # to be - a list/vector/matrix of numbers - and initialise it with some random
        # values
        self.chromosome = np.random.rand(7)
        # 0 = hunter gene -- move towards enemy
        # 1 = scared gene -- move from enemy
        # 2 = friendly gene -- move to friends
        # 3 = wall gene -- move from wall
        # 4 = hungry gene -- move towards enemy
        # 5 = chomp gene -- eat food
        # 6 = exploration -- essentially random movement

    def AgentFunction(self, percepts):
        actions = [0, 0, 0, 0, 0]  # left, up, right, down, eat
        creaturePerc = percepts[:, :, 0]  # creatures. 2,2,0 = player pos. x > 0 = friendly, x < 0 = enemy
        foodPerc = percepts[:, :, 1]  # food. 1 = strawberry, 0 = none
        wallPerc = percepts[:, :, 2]  # walls. 1 = wall, 0 = clear

        #  exploration, random direction = random * explore gene
        actions[np.random.randint(0,4)] += np.random.rand() * self.chromosome[6]

        #  sense percepts
        for row in range(5):
            for col in range(5):
                creature = creaturePerc[row][col] if creaturePerc[row][col] < 0 else 0  # X > 0 = FRIEND. X < 0 = ENEMY
                if creature < 0:  # ENEMY
                    #print("ENEMY")
                    # for some reason works REALLY well without these two on here
                    self.alterActions(row, col, 0, abs(creature), True, actions)  # HUNTER GENE should be true
                    #self.alterActions(row, col, 1, abs(creature), False, actions)  # FLEE GENE

                elif creature > 0:  # FRIEND
                    #print("FRIEND")
                    self.alterActions(row, col, 2, creature, True, actions)  # FRIENDLY GENE

                wall = wallPerc[row][col]
                if wall == 1:  # WALL
                    #print("WALL")
                    self.alterActions(row, col, 3, wall, False, actions)  # WALL GENE

                food = foodPerc[row][col]
                if food == 1:  # FOOD
                    #print("FOOD")
                    self.alterActions(row, col, 4, food, True, actions)  # HUNGRY GENE

        #print(str(actions))
        return actions

    def alterActions(self, col, row, type, percep, towards, actions):
        actionVal = self.chromosome[type] * percep

        if col == 2 and row == 2:  # center
            actions[4] += self.chromosome[5] * percep  # CHOMP GENE
        else:
            if towards:  # move towards percep object
                if self.isLeft(col):  # left
                    actions[0] += actionVal
                else:  # right
                    actions[2] += actionVal

                if self.isDown(row):  # down
                    actions[3] += actionVal
                else:  # up
                    actions[1] += actionVal
            else:  # move from percep object
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
    mutationRate = 0.5  # 0 to 1, representing a percentage -- Default 0.5
    elitismRate = 0.25  # 0 to 1, representing a percentage -- Default 0.25

    printStats = False  # for chromosome stats
    # This function should also return average fitness of the old_population
    N = len(old_population)

    # Fitness for all agents
    fitness = np.zeros((N))
    avgGenes = [0,0,0,0,0,0,0]

    # fitness stats
    food = 0
    kills = 0
    movements = 0
    bounces = 0
    turns = 0

    if printStats: print("\nINDEX: FITNESS ------------------")  # -------------- FITNESS FUNCTION ---
    for n, creature in enumerate(old_population):
        # creature.alive (boolean), creature.turn (int), creature.size (int), creature.strawb_eats (int),
        # creature.enemy_eats (int), creature.squares_visited (int), creature.bounces (int))
        food += creature.strawb_eats
        kills += creature.enemy_eats
        movements += creature.squares_visited
        bounces += creature.bounces
        turns += creature.turn

        """fitness[n] += 50 if creature.alive else (creature.turn * 0.5)
        fitness[n] += creature.strawb_eats * 5
        fitness[n] += creature.enemy_eats * 10
        fitness[n] += creature.squares_visited"""

        fitness[n] += 60 if creature.alive else (creature.turn * 0.55)  # 50 and 0.5
        #fitness[n] += 30 if creature.alive else 0
        fitness[n] += creature.strawb_eats * 7  # 5
        fitness[n] += creature.enemy_eats * 10  # 10
        #fitness[n] += creature.squares_visited  # 1

        if printStats:
            print(str(n) + ": " + str(fitness[n]))
            for i in range(len(creature.chromosome)):
                avgGenes[i] += creature.chromosome[i]

    topIndexes = sorted(range(len(fitness)), key=lambda i: fitness[i])[-int(len(fitness) * elitismRate):]
    topLen = len(topIndexes)

    # ---------------------------------------------------------------------------------
    #print("\n\nfood: " + str(food/N) + ". Kills: " + str(kills/N) + ". Movements: " + str(movements/N) + ". Bounces: " + str(bounces/N) + ". Turns: " + str(turns/N))

    if printStats:
        print("TOP FITNESS INDEXES: " + str(topIndexes))
        print("GENE: AVG ------------------")
        for gene in range(len(avgGenes)):
            avgGenes[gene] = avgGenes[gene] / N
            print("gene" + str(gene) + ": " + str(avgGenes[gene]))

    # -------------------------------------------------------------- NEW POPULATION ---
    new_population = list()
    for n in range(N):

        # Create new creature
        new_creature = MyCreature()
        if n < topLen:  # elitism (keep fittest n of population) -----------------------
            nextTopFittest = topIndexes.pop(0)
            new_creature = old_population[nextTopFittest]

        else :  # else select other ----------------------------------------------------
            parent1 = tournamentParents(old_population, fitness)  # tournament selection
            parent2 = tournamentParents(old_population, fitness)  # tournament selection

            new_creature.chromosome = crossoverChromosome(parent1.chromosome, parent2.chromosome, "point")
            new_creature.chromosome = mutation(mutationRate, new_creature.chromosome)

        new_population.append(new_creature)

    # At the end you need to compute average fitness and return it along with your new population
    avg_fitness = np.mean(fitness)
    return new_population, avg_fitness


# chance = number between 0 and 1 (float)
def mutation(chance, chromosome):
    if np.random.rand() < chance:  # mutation
        chromosome[random.randint(0, len(chromosome)-1)] = np.random.rand()
    return chromosome


# type 1 = random point crossover, type 2 = half/half, type 3 = random multi section crossover
# just like in real life, this is all random
def crossoverChromosome(chromosome1, chromosome2, crossType):
    newChromosome = []
    parents = [chromosome1, chromosome2]

    # random point crossover
    if crossType == "point":
        for i in range(len(chromosome1)):
            newChromosome.append(random.choice(parents)[i])

    return newChromosome


# one parent selected over another based on fitness
def tournamentParents(population, fitness):
    parent1 = population[random.randint(0, len(population)-1)]
    parent2 = population[random.randint(0, len(population)-1)]
    if fitness[population.index(parent1)] > fitness[population.index(parent2)]:
        return parent1
    else:
        return parent2
