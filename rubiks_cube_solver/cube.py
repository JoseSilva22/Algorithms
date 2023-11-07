#https://carlosgrande.me/rubiks-cube-model/
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
#https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html

'''
       Z
       ^
       |
       |
       |-------> Y
      /
    /
    X
'''


from scipy.spatial.transform import Rotation as R
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Rectangle
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import random
    

'''
A Cube contains 28 Blocks (ignoring center)
A Block may contain up to 3 Squares
A Face contains 9 Blocks that should make up to 21 Sqaures
'''

colors_mapping = [
    'green','red','yellow','green','red','green',
    'red','white','green','yellow','green','green',
    'white','green','orange','yellow','green','orange',
    'green','orange','white','red','yellow','red',
    'red','white','yellow','white','orange','yellow',
    'orange','orange','white','blue','red','yellow',
    'blue','red','blue','red','white','blue',
    'yellow','blue','blue','white','blue','orange',
    'yellow','blue','orange','blue','orange','white',
]

face_mapping = {
    "BACK":    (-1, 0), #(value, index of pos tuple)
    "FRONT":   (1, 0), 
    "RIGHT":   (1, 1), 
    "LEFT":    (-1, 1), 
    "TOP":     (1, 2), 
    "BOTTOM":  (-1, 2),
    "X_PLANE": (0, 0),
    "Y_PLANE": (0, 1),
    "Z_PLANE": (0, 2),
}

axis_rotations = {
    "Z0": R.from_quat([               0,                0, np.sin(-np.pi/4), np.cos(-np.pi/4)]), # Z clockwise
    "Z1": R.from_quat([               0,                0,  np.sin(np.pi/4),  np.cos(np.pi/4)]), # Z counterclockwise
    "Y0": R.from_quat([               0, np.sin(-np.pi/4),                0, np.cos(-np.pi/4)]), # Y clockwise
    "Y1": R.from_quat([               0,  np.sin(np.pi/4),                0,  np.cos(np.pi/4)]), # Y counterclockwise
    "X0": R.from_quat([np.sin(-np.pi/4),                0,                0, np.cos(-np.pi/4)]), # X clockwise
    "X1": R.from_quat([ np.sin(np.pi/4),                0,                0,  np.cos(np.pi/4)]), # X counterclockwise
}


class Strategy(ABC):
    @abstractmethod
    def do_algorithm(self, cube):
        pass


class GeneticAlgorithmStrategy(Strategy): 
    def __init__(self, chromo_size = 22, pop_size = 100, generations = 250):
        self.chromo_size = chromo_size
        self.pop_size = pop_size
        self.generations = generations
        
    def init_population(self):
        faces = list(face_mapping.keys())
        pop = [([(random.choice(faces), random.randint(0,1)) for _ in range(self.chromo_size)], 0) for _ in range(self.pop_size)]
        return pop
    
    def fitness(self, indiv, cube):
        cube2 = deepcopy(cube)
        for op in indiv:
            face_name, direction = op[0], op[1]
            cube2.rotate_face(face_name, direction)
        
        return cube2.evaluate()
        
    def do_algorithm(self, cube):
        population = self.init_population()
        population = [(indiv[0], self.fitness(indiv[0], cube)) for indiv in population]
        
        for i in range(self.generations):
            population = self.init_population()
            population = [(indiv[0], self.fitness(indiv[0], cube)) for indiv in population]
            
            print(sorted([indiv[1] for indiv in population], reverse=True)[:3])
            '''
            # parents selection
            mate_pool = sel_parents(populacao)
            # Variation
            # ------ Crossover
            progenitores = []
            for i in  range(0,size_pop-1,2):
                indiv_1= mate_pool[i]
                indiv_2 = mate_pool[i+1]
                filhos = recombination(indiv_1,indiv_2, prob_cross)
                progenitores.extend(filhos) 
            # ------ Mutation
            descendentes = []
            for cromo,fit in progenitores:
                novo_indiv = mutation(cromo,prob_mut, domain,sigma)
                descendentes.append((novo_indiv,fitness_func(novo_indiv)))
            # New population
            populacao = sel_survivors(populacao,descendentes)
            # Evaluate the new population
            populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]     
            '''
        return best_pop(populacao)


class SimulatedAnnealingStrategy(Strategy):
    def __init__(self, temp = 1000, teta = 0.05):
        self.temp = temp
        self.teta = teta
    
    def do_algorithm(self, cube):
        while self.temp != 0:
            cube2 = deepcopy(cube)
            self.temp *= self.teta
        
        
        
class Square:
    def __init__(self, normal, color):
        self.orig_normal = normal
        self.curr_normal = normal
        self.color = color
    
    def __str__(self):
        return f"Color: {self.color}, Normal: {self.curr_normal}"
        
    def check(self):
        return self.orig_normal == self.curr_normal
        

class Block:
    def __init__(self, pos):
        self.pos = pos
        self.squares = []
        
    def add_square(self, square):
        self.squares.append(square)
        
    def check(self):
        score = 0
        for square in self.squares:
            score += square.check()

        return score
    
        
class Cube:
    def __init__(self, strategy):
        self.blocks = []
        self.solver = strategy
    
    def add_block(self, block):
        self.blocks.append(block)    
    
    def get_face(self, face_name):
        val, ix = face_mapping[face_name]
        face = []
        for block in self.blocks:
            if block.pos[ix] == val:
                face.append(block)
        return face
        
    def rotate_face(self, face_name, direction):
        # 0 - clockwise, 1 - counterclockwise
        ix_to_axis = face_mapping[face_name][1] # get ix
        axis = "X"
        if ix_to_axis == 1:
            axis = "Y"
        elif ix_to_axis == 2:
            axis = "Z"
        
        r = axis_rotations[f"{axis}{direction}"]
        face = self.get_face(face_name)
        
        for block in face:
            # rotate block
            block.pos = list(map(int, r.apply(block.pos))) # to fix floating point results
            for square in block.squares:
                #rotate square
                square.curr_normal = list(map(int, r.apply(square.curr_normal))) # to fix floating point results
    
    def shuffle(self, steps=50):
        faces = list(face_mapping.keys())
        
        for _ in range(steps):
            face_name = random.choice(faces)
            self.rotate_face(face_name, random.randint(0,1)) # clockwise or counterclockwise
        
    def solve(self):
        return self.solver.do_algorithm(self)
        
    def evaluate(self):
        # assumes that faces end at the initial positions
        # does not account for cube rotations
        score = 0

        for block in self.blocks:
            score += block.check()
        
        return score
        
        
        
if __name__ == '__main__':
    ga = GeneticAlgorithmStrategy()
    sa = SimulatedAnnealingStrategy()
    cube = Cube(ga)
    
    # ================== PREPARE CUBE =================
    
    count = 0
    for i in range(-1,2):
        for j in range(-1,2):
            for k in range(-1,2):
                # if contains two 0s is a center square
                # if contains one 0 is a border square
                # if contains no 0s is a corner square
                
                zeros = (i,j,k).count(0)
                
                # cube center
                if zeros == 3:
                    continue
                
                block = Block([i,j,k])
                
                if i != 0:
                    block.add_square(Square([i, 0, 0], colors_mapping[count])) 
                    count += 1
                if j != 0:
                    block.add_square(Square([0, j, 0], colors_mapping[count])) 
                    count += 1
                if k != 0:
                    block.add_square(Square([0, 0, k], colors_mapping[count])) 
                    count += 1

                cube.add_block(block)
                
    
    
    # =================== DEBUGGING ===================
    '''
    for block in cube.blocks:
        print(f"\nBlock squares: {len(block.squares)}")
        print(f"Block pos: {block.pos}")
        for square in block.squares:
            print(square)
    
    face_name = "X_PLANE"
    le_face = cube.get_face(face_name)
    
    print(f"\n# blocks in face {face_name}: {len(le_face)}")
    for block in le_face:
        print(f"Block pos: {block.pos}")
        for square in block.squares:
            print(square)
        
    print("Score for solved Cube should be 54")
    print(f"Score: {cube.evaluate()}")
    '''
    # ================ SHUFFLE + SOLVE ================
    
    cube.shuffle()
    print(f"Score: {cube.evaluate()}")
    
    
    fig = plt.figure()  
    ax = fig.add_subplot(projection='3d')
    
    for block in cube.blocks:
        for square in block.squares:
            zdir = "x"
            offset = block.pos[0]
            rect_pos = (block.pos[1], block.pos[2])
            if abs(square.curr_normal[1]) == 1:
                zdir = "y"
                rect_pos = (block.pos[0], block.pos[2])
                offset = block.pos[1]
            elif abs(square.curr_normal[2]) == 1:
                zdir = "z"
                rect_pos = (block.pos[0], block.pos[1])
                offset = block.pos[2]
            p = Rectangle(rect_pos, 1,1, facecolor=square.color, edgecolor='black')
            ax.add_patch(p)
            if offset == 1:
                offset+=1
            art3d.pathpatch_2d_to_3d(p, z=offset, zdir=zdir)
        
    ax.set_xlim(-1, 2)
    ax.set_ylim(-1, 2)
    ax.set_zlim(-1, 2)

    plt.show()
    
    cube.solve()
    