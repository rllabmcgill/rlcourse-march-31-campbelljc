from env import *
from agents import *
from policy import *

import numpy as np
from keras import initializers

if __name__ == '__main__':
    env = BairdCounterexample()
    policy = BairdRandomPolicy()
    
    num_episodes = 1
    num_steps_per_episode = 1000000
        
    agents = [
        TabularQAgent(env, policy, alpha=0.1, gamma=0.99),
        FAQAgent(env, policy, alpha=0.1, gamma=0.99),
        GreedyGQAgent(env, policy, alpha=0.05, beta=0.25, gamma=0.99)
    ]
    
    for j, agent in enumerate(agents):
        agent.load_stats()
        for ep in range(agent.cur_episode, num_episodes):
            print("Agent", j, ", episode", ep)
            agent.new_episode()
            env.reset()
            reward, terminal = 0, False
            while agent.cur_action_this_episode < num_steps_per_episode:   
                agent.state = np.array(env.get_state())
                agent.update(agent.state, reward, terminal)
                action = agent.choose_action(agent.state)
                reward, terminal = env.take_action(action)
                agent.end_turn()
            agent.end_episode()
        agent.save_stats()
