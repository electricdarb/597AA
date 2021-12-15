import random 
import numpy as np
from math import floor

class Qtable():
    def __init__(self, max_resources, blocks_per_resource, num_classes = 3, actions: list = [0, 1]):
        """
        num_classes: int, number of classes in the model
        num_resources: int, maximum amount of resources that can be assigned for a given class
    
        request space: [[0, 1] for i in range(num_classes)]
        """
        self.blocks_per_resource = blocks_per_resource
        self.max_resources = max_resources
        self.actions = actions # accept or deny request 
        num_actions = len(self.actions)
        # Q table, initialized to 0. 
        self.table = np.zeros((*blocks_per_resource, num_classes, num_actions)) 
       
    def update_table(self, s, a, reward, class_num, s_prime, gamma, alpha):
        index_s = self.map_s(s)
        index_s_prime = self.map_s(s_prime)
        self.table[(*index_s, class_num, a)] = self.table[(*index_s, class_num, a)] \
            + alpha * (reward + gamma * \
            np.max(self.table[(*index_s_prime, class_num)]) -  self.table[(*index_s, class_num, a)])

    def best_action(self, state, slice_):
        index_s = self.map_s(state)
        return np.argmax(self.table[(*index_s, slice_)])

    def map_s(self, state):
        return [floor(s / m * b) for s, m, b in zip(state, self.max_resources, self.blocks_per_resource)]

if __name__ == "__main__":
    num_classes = 3 # there are three classes: utilities, automotive, and manufactuting 
    num_resources = 3 # there are three resources: radio, storage, and compute 

    gamma = .9
    epsilon = .7
    alpha = .3

    max_comp = 500
    max_storage = 1e6
    max_radio = 1e6

    max_resources = np.array([max_comp, max_radio, max_storage])

    delta_comp = 2 #cpus
    delta_storage = 1 #gb
    delta_radio = 100 # mbps

    # action space
    actions = [0, 1]

    def take_action(state): # take action and prevent any invalid state from happened 
        delta_state = np.array([delta_comp, delta_radio, delta_storage])
        state_prime = state + delta_state
        over_limit = False
        for i in range(len(state_prime)):
            if state_prime[i] > max_resources[i]:
                over_limit = True
                break

        if over_limit:
            return 0, 0
        else:
            return delta_state, 1

    # init table with 3 classes and 3 resources
    Q = Qtable(max_resources, [5, 5, 5])

    # init events, each event is shape (num_classes, 3)
    delta_t = 1/40  # 40 steps per hour
    
    lambd = [12*delta_t, 8*delta_t, 10*delta_t] # probabilites that an event happens in a time step
    # since delta 
    beta  = 1/(3*delta_t)
    
    timesteps = 10000
    active_resources = []
    droptimes = []


    state = np.array([0, 0, 0]) #init state to 0
    rewards = np.array([1, 2, 4])

    reward_avg = 0
    class_rewards = ([], [], [])

    for t in range(timesteps):
        # request events will happen at random with a uniform prob of lambd[c]
        for class_num, prob in enumerate(lambd): # iterate through classes
            if random.random() < prob: # if there is a request
                # Q learning s
                action = Q.best_action(state, class_num) if random.random() > epsilon else random.choice(actions)
                delta_state, action = take_action(state) if action else (0, 0) # Bradford Gill II
                s_prime = state + delta_state # Take Action
                reward = 0
                # state matiance
                if action:
                    droptimes.append(t + np.random.exponential(beta)) # calculate droptime of resources allocated
                    active_resources.append(delta_state)
                    reward = rewards[class_num]

                Q.update_table(state, action, reward, class_num, s_prime, alpha, gamma)
                state = s_prime # set new state

                reward_avg = reward_avg * .98 + .02 * reward

                # decrease epsilon to become less greedy
                epsilon *= .99
        i = 0 
        while i < len(active_resources):
            if t > droptimes[i]: # same as delta = 3
                state -= active_resources[i] # remove active resoueces from state
                droptimes.pop(i) 
                active_resources.pop(i)
            else:
                i+=1 

        if t%100 == 0:
            print(reward_avg)
        