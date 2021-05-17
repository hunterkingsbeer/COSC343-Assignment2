import numpy as np
import matplotlib.pyplot as plt
import random

playerName = "myAgent"
nPercepts = 75  # This is the number of percepts
nActions = 5    # This is the number of actions
currentGen = 0  # Tracks current generation
fitnessGraph = np.array([])  # Holds the average fitness for each generation
geneGraph = []
actionsGraph = []
highestGraph = np.zeros(7)

generations = 100
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
    mutationRate = 0.2 # 0 to 1, representing a percentage -- Default 0.2-25
    elitismRate = 0.4  # 0 to 1, representing a percentage -- Default 0.4-5

    printStats = True  # for chromosome stats
    # This function should also return average fitness of the old_population
    N = len(old_population)

    # Fitness for all agents
    fitness = np.zeros((N))
    avgGenes = np.zeros(10)
    avgActions = np.zeros(6)
    highestActions = np.zeros(7)



    # --------------------------------------------------------------------------------------- FITNESS FUNCTION ---------
    for n, creature in enumerate(old_population):
        # creature.alive (boolean), creature.turn (int), creature.size (int), creature.strawb_eats (int),
        # creature.enemy_eats (int), creature.squares_visited (int), creature.bounces (int))

        avgActions[0] += creature.strawb_eats
        if highestActions[0] < creature.strawb_eats:
            highestActions[0] = creature.strawb_eats

        avgActions[1] += creature.enemy_eats
        if highestActions[1] < creature.enemy_eats:
            highestActions[1] = creature.enemy_eats

        avgActions[2] += creature.size
        if highestActions[2] < creature.size:
            highestActions[2] = creature.size

        avgActions[3] += creature.turn
        if highestActions[3] < creature.turn:
            highestActions[3] = creature.turn

        avgActions[4] += creature.squares_visited
        if highestActions[4] < creature.squares_visited:
            highestActions[4] = creature.squares_visited

        avgActions[5] += creature.bounces
        if highestActions[5] < creature.bounces:
            highestActions[5] = creature.bounces

        """ JACKS FITNESS
        fitness[n] += creature.turn * 0.5
        fitness[n] += creature.enemy_eats * 9
        fitness[n] += creature.strawb_eats * 6
        fitness[n] += creature.alive 
        """

        """ ORIGINAL FITNESS
        # MAX: 50 + 45 + 30(i guess?) + 30 (i guess?) =
        fitness[n] += 50 if creature.alive else (creature.turn * 0.5)
        fitness[n] += creature.strawb_eats * 5
        fitness[n] += creature.enemy_eats * 10
        fitness[n] += creature.squares_visited 
        """

        #""" NEW FITNESS
        fitness[n] += creature.enemy_eats * 8
        fitness[n] += creature.size * 10
        fitness[n] += 25 if creature.alive else 0
        #"""

        if highestActions[6] < fitness[n]:
            highestActions[6] = fitness[n]
        if printStats:
            for i in range(len(creature.chromosome)):
                avgGenes[i] += creature.chromosome[i]

    topIndexes = sorted(range(len(fitness)), key=lambda i: fitness[i])[-int(len(fitness) * elitismRate):]

    # --------------------------------------------------------------------------------- PRINT STATS

    if printStats:
        print("\n------------------------ AVG ACTIONS ")
        actionNames = ["FOOD | ", "KILLS | ", "SIZE | ",
                       "TURNS | ", "VISITS | ", "BOUNCES | "]
        for action in range(len(avgActions)):
            avgActions[action] = round(avgActions[action] / N, 2)
            print(actionNames[action] + "AVG: " + str(avgActions[action]) + ", HIGHEST: " + str(highestActions[action]))

        print("\n------------------------ AVG GENES ")
        geneNames = ["HUNTER: ", "FLEE: ",
                     "SOCIAL: ", "ANTISOCIAL: ",
                     "INDOORS: ", "OUTDOORS: ",
                     "HUNGRY: ", "FULL: ",
                     "CHOMP: ", "EXPLORE: "]

        for gene in range(len(avgGenes)):
            avgGenes[gene] = round(avgGenes[gene] / N, 2)
            print(geneNames[gene] + str(avgGenes[gene]))

    # ---------------------------------------------------------------------------------- NEW POPULATION
    new_population = list()
    for n in range(N):

        # Create new creature
        new_creature = MyCreature()
        if n < len(topIndexes):  # elitism (keep fittest n of population)
            nextTopFittest = topIndexes.pop(0)
            new_creature = old_population[nextTopFittest]

        else :  # else select breeding
            parent1 = tournamentParents(old_population, fitness)  # tournament selection
            parent2 = tournamentParents(old_population, fitness)  # tournament selection

            new_creature.chromosome = crossoverChromosome(parent1.chromosome, parent2.chromosome, "gene")
            new_creature.chromosome = mutation(mutationRate, new_creature.chromosome)

        new_population.append(new_creature)

    # At the end you need to compute average fitness and return it along with your new population
    avg_fitness = np.mean(fitness)
    graphPlot(avg_fitness, avgGenes, avgActions, highestActions)

    print("\nFITNESS: ")
    return new_population, avg_fitness


# plots my fitness against the generations
def graphPlot(avg_fitness, avgGenes, avgActions, highestActions):
    global fitnessGraph
    global geneGraph
    global highestGraph
    global currentGen
    global generations
    global trainingAgent

    currentGen += 1

    for i in range(len(highestGraph)):
        if highestGraph[i] < highestActions[i]:
            highestGraph[i] = highestActions[i]

    geneGraph.append(avgGenes)
    actionsGraph.append(avgActions)
    fitnessGraph = np.append(fitnessGraph, avg_fitness)

    if currentGen == generations:
        numGens = np.arange(0, generations)  # needs to be an np array

        fig, axs = plt.subplots(2)  # [0] is the fitness graph, [1] is the gene graph

        # ------------------ FITNESS PLOT
        z = np.polyfit(numGens, fitnessGraph, 1)
        p = np.poly1d(z)
        axs[0].plot(numGens, p(numGens), "k-")  # line of best fit
        axs[0].plot(numGens, fitnessGraph)  # line plot

        axs[0].legend(['Line of Best Fit', 'Generations'], title='Lines', bbox_to_anchor=(1.05, 1.05), loc='upper left')
        axs[1].set_xlabel('Generations')
        axs[0].set_ylabel('Fitness')
        axs[0].set_title('Training against ' + str(trainingAgent)
                         + '\nChange in Fitness over ' + str(generations) + ' Generations')

        # ------------------ GENE PLOT
        colors = ['red', 'blue',  # hunter, flee
                  'cyan', 'olive',  # social, antisocial
                  'gray', 'brown',  # indoor, outdoor
                  'green', 'purple',  # hungry, full
                  'pink', 'orange']  # chomp, explore

        geneGraphArr = np.asarray(geneGraph)  # needs to be an np array
        for i in range(10):
            axs[1].plot(numGens, geneGraphArr[:, i], 'tab:' + colors[i])

            axs[1].legend(['Hunter', 'Flee',
                           'Social', 'Antisocial',
                           'Indoors', 'Outdoors',
                           'Hungry', 'Full',
                           'Chomp', 'Explore'], title='Genes', bbox_to_anchor=(1.05, 1.05), loc='upper left')
        axs[1].set_xlabel('Generations')
        axs[1].set_ylabel('Gene Value')
        axs[1].set_title('Change in Genes over ' + str(generations) + ' Generations')

        # ------------------ ACTIONS PLOT
        highestActionStr = ("Highest Actions -\nFood Eaten: " + str(highestGraph[0]) +
                            ", Enemy Eats: " + str(highestGraph[1]) +
                            ", Size: " + str(highestGraph[2]) +
                            ", Turns: " + str(highestGraph[3]) +
                            ", Visits: " + str(highestGraph[4]) +
                            ". Bounces: " + str(highestGraph[5]) +
                            " Highest Fitness: " + str(highestGraph[6]))

        axs[1].annotate(highestActionStr, (0,0), (0, -50), xycoords='axes fraction', textcoords='offset points', va='top')

        """
        colors = ['green', 'red',  # food, kills
                  'blue', 'pink',  # size, turns
                  'orange', 'purple']  # visits, bounces

        actionsGraphArr = np.asarray(actionsGraph)  # needs to be an np array
        for j in range(6):
            if j < 3:  # food, kills and size scaled up 10x to be visually near other values
                axs[2].plot(numGens, actionsGraphArr[:, j]*10, 'tab:' + colors[j])
            elif j == 3:  # turns scaled down 0.1x to be visually near other values
                axs[2].plot(numGens, actionsGraphArr[:, j] * 0.1, 'tab:' + colors[j])
            elif j == 4:
                axs[2].plot(numGens, actionsGraphArr[:, j]*0.5, 'tab:' + colors[j])
            else:
                axs[2].plot(numGens, actionsGraphArr[:, j], 'tab:' + colors[j])

            axs[2].legend(['Food * 10', 'Kills * 10',
                           'Size * 10', 'Turns * 0.1',
                           'Visits * 0.5', 'Bounces'], title='Actions', bbox_to_anchor=(1.05, 1.05), loc='upper left')

        axs[2].set_xlabel('Generations')
        axs[2].set_ylabel('Action Per Creature Value')
        axs[2].set_title('Change in Average Actions Per Creature over ' + str(generations) + ' Generations'
                         '\n(this is more for visual representation than stats)')
        """

        fig.set_size_inches(20, 18, forward=True)

        plt.tight_layout()
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
        for i in range(0, len(chromosome1), 2):
            parent = random.choice(parents)

            newChromosome.append(parent[i])
            newChromosome.append(parent[i+1])

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
