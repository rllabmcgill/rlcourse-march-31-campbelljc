import numpy as np

class BairdRandomPolicy():
    def select_action(self, q_values):
        return np.random.choice([0, 1], p=[1/6, 5/6])
