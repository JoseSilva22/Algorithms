'''
def rotate_face_clockwise(cube, face):
    # Rotate a single face clockwise (90 degrees)
    rotated_face = [cube[face][6], cube[face][3], cube[face][0], cube[face][7], cube[face][4], cube[face][1], cube[face][8], cube[face][5], cube[face][2]]
    new_cube = [list(row) for row in cube]  # Create a copy of the cube
    for i in range(9):
        new_cube[face][i] = rotated_face[i]
    return new_cube

def rotate_face_counterclockwise(cube, face):
    # Rotate a single face counterclockwise (270 degrees)
    return rotate_face_clockwise(rotate_face_clockwise(rotate_face_clockwise(cube, face), face), face)

def rotate_cube_clockwise(cube, layer):
    # Rotate the entire cube clockwise (90 degrees) with respect to the given layer
    new_cube = [list(row) for row in cube]  # Create a copy of the cube
    for i in range(3):
        new_cube[layer][i*3:i*3+3], new_cube[layer+1][i*3:i*3+3], new_cube[layer+2][i*3:i*3+3] = [cube[layer+2][i*3], cube[layer+1][i*3], cube[layer][i*3]], [cube[layer+2][i*3+1], cube[layer+1][i*3+1], cube[layer][i*3+1]], [cube[layer+2][i*3+2], cube[layer+1][i*3+2], cube[layer][i*3+2]]
    return new_cube

def rotate_cube_counterclockwise(cube, layer):
    # Rotate the entire cube counterclockwise (270 degrees) with respect to the given layer
    return rotate_cube_clockwise(rotate_cube_clockwise(rotate_cube_clockwise(cube, layer), layer), layer)

# Define the mapping of faces to their positions in the list of lists
face_mapping = {
    1: 0,  # Top
    2: 1,  # Left
    3: 2,  # Front
    4: 3,  # Right
    5: 4,  # Back
    6: 5   # Down
}

# Perform some example operations:
cube = [[1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6],
        [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6],
        [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]]

# Rotate the top face clockwise
cube = rotate_face_clockwise(cube, face_mapping[1])

# Rotate the left face counterclockwise
cube = rotate_face_counterclockwise(cube, face_mapping[2])

# Rotate the entire cube (layers 0 and 1) clockwise
cube = rotate_cube_clockwise(cube, 0)

# Rotate the entire cube (layers 3 and 4) counterclockwise
cube = rotate_cube_counterclockwise(cube, 3)

# Print the resulting cube
for row in cube:
    print(row)
'''


import random

'''
A Cube contains 28 Blocks (ignoring center)
A Block may contain up to 3 Squares
A Face contains 9 Blocks that should make up to 21 Sqaures
'''

face_mapping = {
    "LEFT": (-1, 0), 
    "RIGHT": (1, 0), 
    "FRONT": (1, 1), 
    "BACK": (-1, 1), 
    "TOP": (1, 2), 
    "BOTTOM": (-1, 2)   
}

class Square:
    def __init__(self, normal):
        self.orig_normal = normal
        self.curr_normal = normal
        
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
    def __init__(self):
        self.blocks = []
    
    def add_block(self, block):
        self.blocks.append(block)    
    
    def get_face(self, mapping):
        val, ix = mapping
        face = []
        for block in self.blocks:
            if block.pos[ix] == val:
                face.append(block)
        return face
        
    def rotate_face(self, direction):
        # 0 - clockwise, 1 - counterclockwise
        pass
    
    def shuffle(self, steps=50):
        faces = list(face_mapping.keys())
        
        for _ in range(steps):
            face_name = random.choice(faces)
            face = self.get_face(face_name)
            self.rotate_face(face, random.randint(0,1)) # clockwise or counterclockwise
        
    def solve(self, strategy):
        pass
        
    def evaluate(self):
        score = 0
        for block in self.blocks:
            score += block.check()
        
        return score
        
        
if __name__ == '__main__':
    cube = Cube()
    
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
                
                block = Block((i,j,k))
                
                if i != 0:
                    block.add_square(Square((i, 0, 0)))
                if j != 0:
                    block.add_square(Square((0, j, 0)))
                if k != 0:
                    block.add_square(Square((0, 0, k)))

                cube.add_block(block)
    
    
    # =================== DEBUGGING ===================
    
    for block in cube.blocks:
        print(f"\nBlock squares: {len(block.squares)}")
        print(f"Block pos: {block.pos}")
        for square in block.squares:
            print(square.curr_normal)
    
    le_face = cube.get_face(face_mapping["LEFT"])
    
    print(f"# blocks in face: {len(le_face)}")
    for block in le_face:
        print(block.pos)
        for square in block.squares:
            print(square.curr_normal)
        
    print("Score for solved Cube should be 54")
    print(f"Score: {cube.evaluate()}")
    assert 54 == cube.evaluate()
    