#https://carlosgrande.me/rubiks-cube-model/
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html

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
# r = R.from_quat([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)]) # Eixo-Z rotation
# r = R.from_quat([0, 0, -np.sin(np.pi/4), -np.cos(np.pi/4)]) # Eixo-Z rotation

from scipy.spatial.transform import Rotation as R
from abc import ABC, abstractmethod
import numpy as np
import random

'''
A Cube contains 28 Blocks (ignoring center)
A Block may contain up to 3 Squares
A Face contains 9 Blocks that should make up to 21 Sqaures
'''


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
    def do_algorithm(self, cube):
        pass


class SimulatedAnnealingStrategy(Strategy):
    def do_algorithm(self, cube):
        pass
        
        
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
        #print(face_name)
        #print(axis)
        #print(r.as_euler('zyx', degrees=True))
        
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
                    block.add_square(Square([i, 0, 0], 'RED')) # fix color calc
                if j != 0:
                    block.add_square(Square([0, j, 0], 'RED')) # fix color calc
                if k != 0:
                    block.add_square(Square([0, 0, k], 'RED')) # fix color calc

                cube.add_block(block)
    
    
    # =================== DEBUGGING ===================
    
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
    
    # ================ SHUFFLE + SOLVE ================
    
    cube.shuffle(1)
    print(f"Score: {cube.evaluate()}")