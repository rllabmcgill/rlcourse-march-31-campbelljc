import numpy as np
from collections import defaultdict

class Env():
    def __init__(self, name, num_actions, num_states, num_features=None):
        self.name = name
        self.num_actions = num_actions
        self.num_states = num_states
        self.num_features = num_states if num_features is None else num_features
        self.state = self.get_initial_state()
    
    def get_initial_state(self):
        raise NotImplementedError
    
    def get_state(self):
        if isinstance(self.state, list):
            return self.state
        else:
            assert 0 <= self.state <= 1
            return [self.state]
    
    def reset(self):
        self.state = self.get_initial_state()
    
    # return (reward, terminal=T/F)
    def take_action(self, action):
        raise NotImplementedError

class BairdCounterexample(Env):
    def __init__(self):
        super().__init__('baird_counterexample', num_actions=2, num_states=6, num_features=7)
    
    def get_initial_state(self):
        state = [0] * self.num_states
        state[-1] = 1
        return state
    
    def get_features(self):
        terminal_state_index = self.num_states-1
        num_features = self.num_states+1 # if 7 states, 8 features per state
        bias_weight_index = num_features-1
        
        solid_action = 0
        dotted_action = 1
        
        features = [[[0 for i in range(num_features)] for j in range(self.num_actions)] for j in range(self.num_states)]
        for state in range(self.num_states):
            features[state][dotted_action][state] = 1 # dotted action: one weight.
            features[state][solid_action][state] = 2 # solid action: 2 weights (w_0 + 2*w_i)
            features[state][solid_action][bias_weight_index] = 1
        features[terminal_state_index][solid_action][terminal_state_index] = 1 # terminal state solid act: 2*w_0 + w_term
        features[terminal_state_index][solid_action][bias_weight_index] = 2
        
        return features
    
    def get_initial_weights(self):
        terminal_state_index = self.num_states-1
        num_features = self.num_states+1 # if 7 states, 8 features per state
        bias_weight_index = num_features-1
        
        solid_action = 0
        dotted_action = 1
        
        weights = [[0.0 for i in range(num_features)] for j in range(self.num_actions)]
        for i in range(num_features):
            weights[solid_action][i] = 1.0 # solid qvs start larger than dotted
            weights[dotted_action][i] = 1.0
        weights[solid_action][terminal_state_index] = 10.0 # transition from terminal>terminal largest
        
        print("Weights (env):\n", np.array(weights))
        
        return weights
    
    def get_initial_qvalues(self):
        terminal_state_index = self.num_states-1
        solid_action = 0
        
        qvals = defaultdict(lambda: 0)
        for i, state in enumerate(np.identity(self.num_states)):
            qvals[tuple(state)] = np.array([1, 1])
            print(qvals[tuple(state)])
            if i == terminal_state_index:
                qvals[tuple(state)][solid_action] = 10
        return qvals
    
    def take_action(self, action):
        # transition to next state..
        assert action in [0, 1]
        if action == 0: # solid
            # first action always goes back to initial state.
            self.state = self.get_initial_state()
        elif action == 1: # dotted
            # second action goes to any state!=last with uniform prob.
            state_index = np.random.choice([i for i in range(self.num_states-1)])
            self.state = [0] * self.num_states
            self.state[state_index] = 1
        
        return 0, False # never any rewards
    
class BairdCounterexample2(Env):
    def __init__(self):
        super().__init__('baird_counterexample2', num_actions=1, num_states=7)
    
    def get_initial_state(self):
        state_index = np.random.choice([1, 2, 3, 4, 5, 6])
        state = [0] * 7
        state[state_index] = 1
        return state
    
    def take_action(self, action):
        # transition to next state..
        assert action in [0, 1]
        self.state = [0] * 7
        self.state[0] = 1
        return 0, False # never any rewards
