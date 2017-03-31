import os, math
import numpy as np
import dill, pickle
from graphs import Episode
from functools import partial
from collections import defaultdict
from fileio import append, write_list, read_list

class Agent():
    def __init__(self):
        self.cur_episode = 0
        self.cur_action = 0
        self.game_records = []
        self.load_stats()
        
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
    
    def load_stats(self):
        if os.path.exists(self.savedir+"stats.txt"):
            self.cur_episode, self.cur_action = read_list(self.savedir+"stats")
        if os.path.exists(self.savedir+"records.dll"):
            with open(self.savedir+"records.dll", 'rb') as finput:
                self.game_records = dill.load(finput)
    
    def save_stats(self):
        write_list([self.cur_episode, self.cur_action], self.savedir+"stats")
        with open(self.savedir+"records.dll", 'wb') as output:
            dill.dump(self.game_records, output)
    
    def new_episode(self):
        self.total_reward = 0
        self.cur_action_this_episode = 0
    
    def end_turn(self):
        self.cur_action += 1
        self.cur_action_this_episode += 1
    
    def end_episode(self):
        self.game_records.append(Episode(self.cur_episode, self.cur_action_this_episode, self.total_reward))
        self.cur_episode += 1

class QAgent(Agent):
    def __init__(self, env, policy, alpha, gamma, name):
        self.policy = policy
        self.state_counts = defaultdict(int) # count of state, action tuples        
        self.alpha = alpha
        self.gamma = gamma
        self.num_actions = env.num_actions
        self.num_states = env.num_states
        
        self.savedir = '../results/model_' + env.name + '_' + name + '/'
        super().__init__()
    
    def load_stats(self):
        super().load_stats()
        
        if os.path.exists(self.savedir+"params.txt"):
            self.alpha, self.gamma = read_list(self.savedir+"params")
    
    def save_stats(self):
        super().save_stats()
        
        write_list([self.alpha, self.gamma], self.savedir+"params")
    
    def new_episode(self):
        super().new_episode()
        self.state = None
        self.prev_state = None
        self.last_action = None
        
    def end_turn(self):
        self.prev_state = self.state
        super().end_turn()
        
    def choose_action(self, state):
        raise NotImplementedError
        
    def update(self, state, reward, terminal):
        raise NotImplementedError

class TabularQAgent(QAgent):
    def __init__(self, env, policy, alpha, gamma, name='QLearningTabular'):
        self.qvals = defaultdict(int)
        if 'get_initial_qvalues' in dir(env):
            self.qvals = env.get_initial_qvalues()
        
        super().__init__(env, policy, alpha, gamma, name)
        
    def choose_action(self, state):
        qvals = self.qvals[tuple(state)]
        action = self.policy.select_action(q_values=qvals)
        self.last_action = action
        #print("acts for state", state, ":", qvals)
        return action
    
    def end_turn(self):
        super().end_turn()
        summed_qvals = 0
        for state in np.identity(self.num_states):
            summed_qvals += max(self.qvals[tuple(state)])
        append(summed_qvals, self.savedir+"/summed_qvals")
    
    def update(self, state, reward, terminal):
        if self.cur_action_this_episode == 0:
            return # skip initial state (haven't chosen an action yet)
        
        assert self.last_action is not None
        assert self.prev_state is not None
        
        old_state = self.prev_state
        action = self.last_action
        new_state = state
        
        self.total_reward += reward
        
        if terminal:
            update = reward # tt = rr
        else: # non-terminal state
            newQ = self.qvals[tuple(new_state)][:]
            maxQ = np.max(newQ) # maxQ = max_a' Q(ss', a') for all a' in A
            update = (reward + (self.gamma * maxQ)) # tt = rr + gamma * max_a' Q(ss', a') for all a' in A
        
        cur_qval = self.qvals[tuple(old_state)][action]
        alpha = 1/((self.state_counts[(tuple(old_state), action)] + 1) ** self.alpha)
        self.qvals[tuple(old_state)][action] = ((1-alpha)*cur_qval) + (alpha*update)
                
        self.state_counts[(tuple(old_state), action)] += 1

class FAQAgent(QAgent):
    def __init__(self, env, policy, alpha, gamma, name='FAQAgent'):
        super().__init__(env, policy, alpha, gamma, name)
        self.num_features = env.num_features
        self.weights = [[0 for i in range(self.num_features)] for j in range(self.num_actions)]
        if 'get_initial_weights' in dir(env):
            self.weights = env.get_initial_weights()
        self.features = env.get_features()
        print("Feats\n", np.array(self.features), "\nweights:\n", np.array(self.weights))
    
    def choose_action(self, state):
        action = self.policy.select_action(q_values=self.predict(state))
        self.last_action = action
        return action
    
    def end_turn(self):
        super().end_turn()
        summed_qvals = 0
        for state in np.identity(self.num_states):
            summed_qvals += self.predict(state)
        append(summed_qvals, self.savedir+"/summed_qvals")
    
    def predict(self, state):
        maxQ = -np.inf
        for i in range(self.num_actions):
            maxQ = max(maxQ, self.get_qvalue(state, i))
        return maxQ
    
    def get_qvalue(self, state, action):
        state_features = np.transpose(np.dot(np.transpose(self.features), state))
        return np.dot(self.weights[action][:], state_features[action])
    
    def update(self, state, reward, terminal):
        if self.cur_action_this_episode == 0:
            return # skip initial state (haven't chosen an action yet)
        
        assert self.last_action is not None
        assert self.prev_state is not None
        
        state = self.prev_state
        action = self.last_action
        next_state = state
        
        self.total_reward += reward
        
        td_error = reward + (self.gamma * self.predict(next_state)) - self.get_qvalue(state, action)
        state_features = np.transpose(np.dot(np.transpose(self.features), state))
        
        self.weights += self.alpha * td_error * state_features

class GreedyGQAgent(QAgent):
    def __init__(self, env, policy, alpha, beta, gamma, name='GreedyGQAgent'):
        super().__init__(env, policy, alpha, gamma, name)
        self.num_features = env.num_features
        self.thetas  = np.array([[0.0 for i in range(self.num_features)] for j in range(self.num_actions)]).reshape((self.num_features*self.num_actions,))
        self.weights = np.array([[0.0 for i in range(self.num_features)] for j in range(self.num_actions)]).reshape((self.num_features*self.num_actions,))
        if 'get_initial_weights' in dir(env):
            self.thetas = np.array(env.get_initial_weights()).reshape((self.num_features*self.num_actions,))
        self.features = np.array(env.get_features()).reshape((self.num_states,self.num_features*self.num_actions))
        self.beta = beta
        print("Feats\n", np.array(self.features), "\nweights:\n", np.array(self.thetas))
    
    def choose_action(self, state):
        action = self.policy.select_action(q_values=self.predict(state))
        self.last_action = action
        return action
    
    def end_turn(self):
        super().end_turn()
        summed_qvals = 0
        for state in np.identity(self.num_states):
            summed_qvals += self.predict(state)
        append(summed_qvals, self.savedir+"/summed_qvals")
    
    def predict(self, state):
        maxQ = -np.inf
        for i in range(self.num_actions):
            maxQ = max(maxQ, self.get_qvalue(state, i))
        return maxQ
    
    def get_qvalue(self, state, action):
        state_features = np.transpose(np.dot(np.transpose(self.features), state))
        return np.dot(self.thetas.reshape((self.num_actions, self.num_features))[action][:], state_features.reshape((self.num_actions, self.num_features))[action])
    
    def update(self, state, reward, terminal):
        if self.cur_action_this_episode == 0:
            return # skip initial state (haven't chosen an action yet)
        
        assert self.last_action is not None
        assert self.prev_state is not None
        
        state = self.prev_state
        action = self.last_action
        next_state = state
        
        self.total_reward += reward
        
        # ref: gq(lambda) ref guide and java implementation
        #      https://groups.google.com/forum/?hl=en#!topic/rl-list/xC7CEvWLUjc
        
        state_features = np.transpose(np.dot(np.transpose(self.features), state)) # dim: (numacts, numfeats)
        next_state_features = np.transpose(np.dot(np.transpose(self.features), next_state)) # dim: (numacts, numfeats)
        
        td_error = reward + (self.gamma * np.dot(np.transpose(self.thetas), next_state_features)) - np.dot(np.transpose(self.thetas), state_features)
        self.thetas += self.alpha * ((td_error * state_features) - (self.gamma * (np.dot(np.transpose(self.weights), state_features)) * next_state_features))
        self.weights += self.beta * ((td_error * state_features) - (np.dot(np.transpose(self.weights), state_features) * state_features))
        