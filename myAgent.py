import numpy as np
import matplotlib.pyplot as plt
import random

playerName = "myAgent"
nPercepts = 75  # This is the number of percepts
nActions = 5    # This is the number of actions
currentGen = 0  # Tracks current generation
fitnessGraph = np.array([])  # Holds the average fitness for each generation
geneGraph = []


generations = 10
trainingSchedule = [("random", generations)]

class MyCreature:
    def __init__(self):

        self.chromosome = np.random.rand(10)

        self.hunter = 0
        self.flee = 1
        # 0 = hunter gene -- move towards enemy
        # 1 = flee gene -- move from enemy

        self.social = 2
        self.antisocial = 3
        # 2 = social gene -- move to friends
        # 3 = antisocial gene -- move from friends

        self.indoors = 4
        self.outdoors = 5
        # 4 = indoors gene -- move to wall
        # 5 = outdoors gene -- move from wall

        self.hungry = 6
        self.full = 7
        # 6 = hungry gene -- move towards food
        # 7 = full gene -- move from food

        self.chomp = 8
        self.exploration = 9
        # 8 = chomp gene -- eat food
        # 9 = exploration -- essentially random movement

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
                    self.alterActions(row, col, self.hunter, abs(creature), True, actions)  # HUNTER GENE
                    self.alterActions(row, col, self.flee, abs(creature), False, actions)  # FLEE GENE

                elif creature > 0:  # FRIEND

                    # print("FRIEND")
                    self.alterActions(row, col, self.social, creature, True, actions)  # FRIENDLY GENE
                    self.alterActions(row, col, self.antisocial, creature, True, actions)  # ANTISOCIAL GENE

                wall = wallPerc[row][col]
                if wall == 1:  # WALL

                    # print("WALL")
                    self.alterActions(row, col, self.indoors, wall, True, actions)  # INDOORS GENE
                    self.alterActions(row, col, self.outdoors, wall, False, actions)  # OUTDOORS GENE

                food = foodPerc[row][col]
                if food == 1:  # FOOD

                    # print("FOOD")
                    self.alterActions(row, col, self.hungry, food, True, actions)  # HUNGRY GENE
                    self.alterActions(row, col, self.full, food, False, actions)  # FULL GENE

        return actions

    def alterActions(self, col, row, type, percep, towards, actions):
        actionVal = self.chromosome[type] * percep

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
    mutationRate = 0.1 # 0 to 1, representing a percentage -- Default 0.5
    elitismRate = 0.25  # 0 to 1, representing a percentage -- Default 0.25

    printStats = True  # for chromosome stats
    # This function should also return average fitness of the old_population
    N = len(old_population)

    # Fitness for all agents
    fitness = np.zeros((N))
    avgGenes = np.zeros(10)

    # fitness stats
    food = 0
    kills = 0
    movements = 0
    bounces = 0
    turns = 0
    size = 0

    # --------------------------------------------------------------------------------------- FITNESS FUNCTION ---------
    for n, creature in enumerate(old_population):
        # creature.alive (boolean), creature.turn (int), creature.size (int), creature.strawb_eats (int),
        # creature.enemy_eats (int), creature.squares_visited (int), creature.bounces (int))
        food += creature.strawb_eats
        kills += creature.enemy_eats
        movements += creature.squares_visited
        bounces += creature.bounces
        turns += creature.turn
        size += creature.size

        """ JACKS FITNESS
        fitness[n] += creature.turn * 0.5
        fitness[n] += creature.enemy_eats * 9
        fitness[n] += creature.strawb_eats * 6
        fitness[n] += creature.alive 
        """


        # THE ORIGINAL 
        # MAX: 50 + 45 + 30(i guess?) + 30 (i guess?) =
        fitness[n] += 50 if creature.alive else (creature.turn * 0.5)
        fitness[n] += creature.strawb_eats * 5
        fitness[n] += creature.enemy_eats * 10
        fitness[n] += creature.squares_visited

        """
        fitness[n] += creature.enemy_eats * 5
        fitness[n] += creature.size * 10
        fitness[n] += creature.turn * 0.2
        fitness[n] += 25 if creature.alive else 0"""

        if printStats:
            for i in range(len(creature.chromosome)):
                avgGenes[i] += creature.chromosome[i]

    topIndexes = sorted(range(len(fitness)), key=lambda i: fitness[i])[-int(len(fitness) * elitismRate):]
    topLen = len(topIndexes)

    # ---------------------------------------------------------------------------------
    print("\n\nAVG STATS ---------------"
          "\nFood: " + str(round(food/N, 2)) +
          ", Kills: " + str(round(kills/N, 2)) +
          ", Movements: " + str(round(movements/N, 2)) +
          ",\nBounces: " + str(round(bounces/N, 2)) +
          ", Turns: " + str(round(turns/N, 2)) +
          ", Size: " + str(round(size/N, 2)))

    if printStats:
        geneNames = ["HUNTER: ", "FLEE: ",
                     "SOCIAL: ", "ANTISOCIAL: ",
                     "INDOORS: ", "OUTDOORS: ",
                     "HUNGRY: ", "FULL: ",
                     "CHOMP: ", "EXPLORE: "]
        print("\nAVG GENES ------------------")
        for gene in range(len(avgGenes)):
            avgGenes[gene] = avgGenes[gene] / N
            print(geneNames[gene] + str(round(avgGenes[gene], 2)))

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
    graphPlot(avg_fitness, avgGenes)

    print("\nFITNESS: ")
    return new_population, avg_fitness


# plots my fitness against the generations
def graphPlot(avg_fitness, avgGenes):
    global currentGen
    global fitnessGraph
    global geneGraph
    global generations

    currentGen += 1

    geneGraph.append(avgGenes)
    fitnessGraph = np.append(fitnessGraph, avg_fitness)

    if currentGen == generations:
        numGens = np.arange(0, generations)  # needs to be an np array

        fig, axs = plt.subplots(2)  # [0] is the fitness graph, [1] is the gene graph

        # ------ FITNESS PLOT
        z = np.polyfit(numGens, fitnessGraph, 1)
        p = np.poly1d(z)
        axs[0].plot(numGens, p(numGens), "k-")  # line of best fit
        axs[0].plot(numGens, fitnessGraph)  # line plot

        axs[0].sharex(axs[1])
        axs[0].set_ylabel('Fitness')
        axs[0].set_title('Change in Fitness over ' + str(generations) + ' Generations')

        # ------ GENE PLOT
        geneGraphArr = np.asarray(geneGraph)  # needs to be an np array
        for i in range(10):
            axs[1].plot(numGens, geneGraphArr[:, i])
        axs[1].set_xlabel('Generations')
        axs[1].set_ylabel('Gene Value')
        axs[1].set_title('Change in Genes over ' + str(generations) + ' Generations')

        plt.show()


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

    # crosses entire genes (keeps the pair for/against genes)
    elif crossType == "gene":
        for i in range(len(chromosome1)):
            pass

    elif crossType == "half":
        for i in range(len(chromosome1)):
            if i > len(chromosome1)/2:
                newChromosome.append(chromosome1[i])
            else:
                newChromosome.append(chromosome2[i])

    return newChromosome


# one parent selected over another based on fitness
def tournamentParents(population, fitness):
    parent1 = population[random.randint(0, len(population)-1)]
    parent2 = population[random.randint(0, len(population)-1)]
    if fitness[population.index(parent1)] > fitness[population.index(parent2)]:
        return parent1
    else:
        return parent2
