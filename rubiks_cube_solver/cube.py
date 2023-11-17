#https://scholar.google.com/scholar_url?url=https://www.academia.edu/download/82983308/IJEME-V8-N1-1.pdf&hl=en&sa=T&oi=gsb-gga&ct=res&cd=0&d=12834093376843814871&ei=-oZXZf6CI4v0mgHazZuwDA&scisig=AFWwaeYgNtJA8akNT4sMnor_CCkQ
#https://carlosgrande.me/rubiks-cube-model/
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
#https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html


from scipy.spatial.transform import Rotation as R
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import random
import time

from strategies.geneticAlgorithm import GeneticAlgorithmStrategy
from strategies.simulatedAnnealing import SimulatedAnnealingStrategy


'''
A Cube contains 28 Blocks (ignoring center)
A Block may contain up to 3 Squares
A Face contains 9 Blocks that should make up to 21 Squares

       Z
       ^
       |
       |
       |-------> Y
      /
     /
    X
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
        moves= []
        for _ in range(steps):
            face_name = random.choice(faces)
            direction = random.randint(0,1)
            moves.append((face_name, 0 if direction == 1 else 1))
            self.rotate_face(face_name, direction) # clockwise or counterclockwise
        
        moves.reverse()
        return (moves, 0)
        
    def solve(self):
        return self.solver.do_algorithm(self, face_mapping)
        
    def evaluate(self):
        # assumes that faces end at the initial positions
        # does not account for cube rotations... (yet!)
        score = 0

        for block in self.blocks:
            score += block.check()
        
        return 54 - score
        
        
        
if __name__ == '__main__':
    ga = GeneticAlgorithmStrategy(generations = 10) # <-- small num of gen just to test, should be much higher
    #sa = SimulatedAnnealingStrategy()
    
    cube = Cube(ga)
    
    # ================== PREPARE CUBE =================
    
    color_ix = 0
    for i in range(-1,2):
        for j in range(-1,2):
            for k in range(-1,2):
                # if contains two 0s is a center square
                # if contains one 0 is a border square
                # if contains no 0s is a corner square
                
                zeros = (i,j,k).count(0)
                
                # center cube -> ignore
                if zeros == 3:
                    continue
                
                block = Block([i,j,k])
                
                if i != 0:
                    block.add_square(Square([i, 0, 0], colors_mapping[color_ix])) 
                    color_ix += 1
                if j != 0:
                    block.add_square(Square([0, j, 0], colors_mapping[color_ix])) 
                    color_ix += 1
                if k != 0:
                    block.add_square(Square([0, 0, k], colors_mapping[color_ix])) 
                    color_ix += 1

                cube.add_block(block)
                
    
    
    # =================== DEBUGGING ===================
    '''
    for block in cube.blocks:
        print(f"\nBlock squares: {len(block.squares)}")
        print(f"Block pos: {block.pos}")
        for square in block.squares:
            print(square)
    
    face_name = "X_PLANE"
    _face = cube.get_face(face_name)
    
    print(f"\n# blocks in face {face_name}: {len(_face)}")
    for block in _face:
        print(f"Block pos: {block.pos}")
        for square in block.squares:
            print(square)
        
    print("Score for solved Cube should be 54")
    print(f"Score: {cube.evaluate()}")
    '''
    # ==================== SHUFFLE ====================
    
    sol = cube.shuffle(10)
    print(f"Cube shuffled! \nCurrent score: {cube.evaluate()} \n0 is the optimal solution\n")
    
    # =================== DRAW CUBE ===================
    
    plt.ion()
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

    #plt.show()
    
    # ===================== SOLVE =====================
    
    sol = cube.solve()
    
    # ================ ANIMATE SOLUTION ===============
    
    for step in sol[0]:
        
        cube.rotate_face(step[0], step[1])
        
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
                
        # drawing updated values
        fig.canvas.draw()
     
        # This will run the GUI event
        # loop until all UI events
        # currently waiting have been processed
        fig.canvas.flush_events()
     
        time.sleep(1)
        
    input()
    