from operator import itemgetter
from copy import deepcopy
import random

from .strategy import Strategy

'''
Next steps:
  - Inject fitness, variation, mutation and selection functions
  - Fitness function evaluate slices of the individuals (solution may be contained in the phenotype)
  - Fitness function contemplate cube rotations (i.e. cube solved but is upside down)
'''

class GeneticAlgorithmStrategy(Strategy): 
    def __init__(self, chromo_size = 40, pop_size = 100, generations = 10):
        self.chromo_size = chromo_size
        self.pop_size = pop_size
        self.generations = generations
        
    def init_population(self, face_mapping):
        faces = list(face_mapping.keys())
        pop = [([(random.choice(faces), random.randint(0,1)) for _ in range(self.chromo_size)], 0) for _ in range(self.pop_size)]
        return pop
    
    def fitness(self, indiv, cube):
        cube2 = deepcopy(cube)
        for op in indiv:
            face_name, direction = op[0], op[1]
            cube2.rotate_face(face_name, direction)
        
        return cube2.evaluate()
    
    # swap mutation
    def muta_cromo(self, cromo, prob_muta):
        if prob_muta < random.random():
            comp = len(cromo) - 1
            copia = cromo[:]
            i = random.randint(0, comp)
            j = random.randint(0, comp)
            while i == j:
                i = random.randint(0, comp)
                j = random.randint(0, comp)
            copia[i], copia[j] = copia[j], copia[i]
            return copia
        else:
            return cromo
    
    # crossover
    def one_point_cross(self, indiv_1, indiv_2,prob_cross):
        value = random.random()
        if value < prob_cross:
            cromo_1 = indiv_1[0]
            cromo_2 = indiv_2[0]
            pos = random.randint(0,len(cromo_1))
            f1 = cromo_1[0:pos] + cromo_2[pos:]
            f2 = cromo_2[0:pos] + cromo_1[pos:]
            return ((f1,0),(f2,0))
        else:
            return (indiv_1,indiv_2)
            
    # Tournament Selection
    def tournament(self, pop):
        size_pop = len(pop)
        mate_pool = []
        for i in range(size_pop):
            winner = self.tour(pop,3)
            mate_pool.append(winner)
        return mate_pool    
        
    def tour(self, population,size):
        pool = random.sample(population, size)
        pool.sort(key=itemgetter(1))
        return pool[0]  
        
    def elitism(self, parents,offspring):
        size = len(parents)
        comp_elite = int(size* 0.1)
        offspring.sort(key=itemgetter(1))
        parents.sort(key=itemgetter(1))
        new_population = parents[:comp_elite] + offspring[:size - comp_elite]
        return new_population
    
    def do_algorithm(self, cube, face_mapping):
        population = self.init_population(face_mapping)
        population = [(indiv[0], self.fitness(indiv[0], cube)) for indiv in population]
        size_pop = len(population)
        
        for i in range(self.generations):
            print(f"\nGeneration: {i+1}")
            # population = self.init_population()
            # population = [(indiv[0], self.fitness(indiv[0], cube)) for indiv in population]
            # print(sorted([indiv[1] for indiv in population], reverse=True)[:3])
            
            # parents selection
            mate_pool = self.tournament(population)
            # Variation
            # ------ Crossover
            progenitors = []
            for i in  range(0,size_pop-1,2):
                indiv_1 = mate_pool[i]
                indiv_2 = mate_pool[i+1]
                sons = self.one_point_cross(indiv_1,indiv_2, 0.8)
                progenitors.extend(sons) 
            # ------ Mutation
            descendants = []
            for cromo,fit in progenitors:
                novo_indiv = self.muta_cromo(cromo,0.2)
                descendants.append((novo_indiv, self.fitness(novo_indiv, cube)))
            # New population
            population = self.elitism(population,descendants)
            # Evaluate the new population
            population = [(indiv[0], self.fitness(indiv[0], cube)) for indiv in population]     
            
            print(f"Fitness: {self.best_pop(population)[1]}")
            
        return self.best_pop(population)
        
    def best_pop(self, pop):
        pop.sort(key=itemgetter(1))
        return pop[0]
