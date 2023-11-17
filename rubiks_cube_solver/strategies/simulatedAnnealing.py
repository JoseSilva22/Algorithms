from .strategy import Strategy


class SimulatedAnnealingStrategy(Strategy):
    def __init__(self, temp = 1000, teta = 0.05):
        self.temp = temp
        self.teta = teta
    
    def do_algorithm(self, cube):
        while self.temp != 0:
            cube2 = deepcopy(cube)
            
            # TO-DO
            
            self.temp *= self.teta
        