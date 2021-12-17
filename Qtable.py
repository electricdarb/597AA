import random 
import numpy as np
from math import floor

class Qtable():
    def __init__(self, max_resources, blocks_per_resource, num_classes = 3, actions: list = [0, 1]):
        """
        args:
            max_resources: list with each cell specifying resource capacity of each resourse
            blocks_per_resource: how many blocks to break resources up into, make resources discrete
            num_classes: number or classes or slices 
            actions: list of possible actions 

        self.table:  a q table that has an element for every reource, class, action combination
        """
        self.blocks_per_resource = blocks_per_resource
        self.max_resources = max_resources
        self.actions = actions # accept or deny request 
        num_actions = len(self.actions)
        self.table = np.zeros((*blocks_per_resource, num_classes, num_actions)) 
       
    def update_table(self, s, a, reward, class_num, s_prime, gamma, alpha):
        index_s = self.map_s(s)
        index_s_prime = self.map_s(s_prime)
        # update table using bellman equation 
        # bellman eq: Q(s, a) = Q(s, a) + alpha * (r + gamma * max(Q(s', a) - Q(s, a)))
        self.table[(*index_s, class_num, a)] = self.table[(*index_s, class_num, a)]  \
            + alpha * (reward + gamma * np.max(self.table[(*index_s_prime, class_num)])\
            - self.table[(*index_s, class_num, a)])

    def best_action(self, state, class_num):
        index_s = self.map_s(state)
        return np.argmax(self.table[(*index_s, class_num)])

    def map_s(self, state):
        return [min(floor(s / m * b), b - 1) for s, m, b in zip(state, self.max_resources, self.blocks_per_resource)]

if __name__ == "__main__":
    #### init model variables ####
    num_classes = 3 # there are three classes: utilities, automotive, and manufactuting 
    num_resources = 3 # there are three resources: radio, storage, and compute 

    max_comp = 10 # total computation resources, unit: CPU
    max_storage = 5 # total storage resources, unit: GB
    max_radio = 500 # total radio resources, unit: Mbps

    max_resources = np.array([max_comp, max_radio, max_storage])

    delta_comp = 2 # CPU
    delta_storage = 1 # GB
    delta_radio = 100 # Mbps

    actions = [0, 1] # action space 

    rewards = np.array([1, 2, 4])
    state = np.array([0, 0, 0]) # init state to 0 for all resources

    def take_action(state): # take action and prevent any invalid state from happened 
        delta_state = np.array([delta_comp, delta_radio, delta_storage])
        #delta_state = np.array([random.uniform(l, h) for l, h in [(1.8, 2.2), (90, 110), (.9, 1.1)]])
        state_prime = state + delta_state
        over_limit = False
        for i in range(len(state_prime)):
            if state_prime[i] > max_resources[i]:
                over_limit = True
                break
        if over_limit:
            return 0, False
        else:
            return delta_state, True

    #### init Q learning  hyper params ####
    gamma = .99 # doscount factor, range (0, 1), higher value -> future is valued higher 
    epsilon = .3 # greedy search factor, range(0, 1), higher value -> more random choices are made
    alpha = .001 # learning rate

    Q = Qtable(max_resources, [5, 5, 5]) # init Q table with 5 state blocks for each resource

    #### init time values ####
    timesteps = int(1e7) # total timesteps to take
    delta_t = 1/40  # 40 steps per hour
    
    lambd = [12 * delta_t, 8 * delta_t, 10 * delta_t] # probabilites that an event happens in a time step
    beta  = 1 / (3 * delta_t) # define beta for expontial distrobution for release time

    active_resources = [] # queue to store active resources in so they can be removed from state when dropped
    droptimes = [] # queue to store droptimes in so the active resources can be removed from 

    reward_avg = .8
    class_rewards = ([], [], [])

    for t in range(timesteps):
        reward = 0
        # request events will happen at random with a uniform prob of lambd[c]
        for class_num, prob in enumerate(lambd): # iterate through classes
            if random.random() < prob: # if there is a request
                action = Q.best_action(state, class_num) if random.random() > epsilon else random.choice(actions)
                delta_state, taken = take_action(state) if action else (0, 0) 
                s_prime = state + delta_state # Take Action
                reward = 0 # set default reward to 0
                if taken:
                    droptimes.append(t + np.random.exponential(beta)) # calculate droptime of resources allocated
                    active_resources.append(delta_state)
                    ###### Write rewards here, could be a function that takes in class_num and weather it is a fog node or not
                    reward = rewards[class_num]
                Q.update_table(state, action, reward, class_num, s_prime, alpha, gamma)
                state = s_prime # set new stat
                reward_avg = reward_avg * .9999 + .0001 * reward

        i = 0 
        while i < len(active_resources):
            if t > droptimes[i]: # same as delta = 3
                state -= active_resources[i] # remove active resoueces from state
                droptimes.pop(i) 
                active_resources.pop(i)
            else:
                i += 1 
        
        if t % 1000000 == 0:
            print(reward_avg, ' ', epsilon)
            #epsilon *= .9975




        