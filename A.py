#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 12:48:49 2023

@author: enrico
"""

import random
from deap import base, creator, tools

creator.create("FitnessMin", base.Fitness, weights = (-1.0, ))
creator.create("Individual", list, fitness = creator.FitnessMin)

IND_SIZE = 2

toolbox = base.Toolbox()
toolbox.register("attribute", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attribute, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    return sum(individual),

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu = 0, sigma = 1, indpb = 0.1)
toolbox.register("select", tools.selTournament, tournsize = 3)
toolbox.register("evaluate", evaluate)

def main():
    pop = toolbox.population(n = 50)
    print(f"pop at beginning: \n {pop}")
    CXPB, MUTPB, NGEN = 0.5, 0.2, 40
    
    # EVALUATE THE ENTIRE POPULATION
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    for g in range(NGEN):
        # SELECT THE NEXT GENERATION INDIVIDUALS
        offspring = toolbox.select(pop, len(pop))
        # CLONE THE SELECTED INDIVIDUALS
        offspring = list(map(toolbox.clone, offspring))
        
        # APPLY CROSSOVER AND MUTATION ON THE OFFSPRING
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # EVALUTATE THE INDIVIDUALS WITH AN INVALID FITNESS
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # THE POPULATION IS ENTIRELY REPLACED BY THE OFFSPRING
        pop[:] = offspring
        
    return pop

print(f"\n \n \n \n \n \n pop at end: \n {main()}")