import os
from env import *
from fileio import *
import seaborn as sns
import matplotlib.pyplot as plt 
from collections import namedtuple

Episode = namedtuple('Episode', 'game_id num_actions total_reward')

if __name__ == '__main__':
    env = BairdCounterexample()
    
    plt.figure()
    plt.ylabel('Summed q-values')
    axes = plt.gca()
    axes.set_ylim([0, 500])
    #axes.set_xlim([0, 2000])
    axes.set_xscale('log')
    #axes.set_yscale('log')
    
    for subdir in get_immediate_subdirectories('../results'):
        if env.name not in subdir:
            continue
        model_name = subdir[len(env.name)+7:]
    
        # load the results for this model
        records = read_line_list("../results/" + subdir+"/summed_qvals")
        plt.plot([i for i in range(len(records))], records, label=model_name)

    # graph the avg total reward
    if not os.path.exists('../figs'):
        os.makedirs('../figs')

    plt.title(env.name + " (summed q-values over time)")
    plt.legend()
    plt.savefig('../figs/' + env.name + '_avgtotalreward.png')