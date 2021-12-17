import numpy as np
from Qtable import Qtable, take_action
import random
import matplotlib.pyplot as plt

q_table = np.load("qtable.npy")
optimum = np.load("optimum.npy")

state = np.array([0,0,0])
timesteps = int(1e5) # total timesteps to take
delta_t = 1/40  # 40 steps per hour

lambd = [12 * delta_t, 8 * delta_t, 10 * delta_t] # probabilites that an event happens in a time step
beta  = 1 / (3 * delta_t) # define beta for expontial distrobution for release time

active_resources = [] # queue to store active resources in so they can be removed from state when dropped
droptimes = [] # queue to store droptimes in so the active resources can be removed from 

class_rewards = ([], [], [])

delta_comp = 2 # CPUs
delta_storage = 1 # GB
delta_radio = 100 # Mbps
delta_res = np.array([delta_comp, delta_radio, delta_storage])

max_res = 12*delta_res

# Initialize Q-table
Q = Qtable(max_res, [5, 5, 5])
Q.table = q_table

rewards = np.array([1, 2, 4])
num_requests = 0
avg_rew_per_req = [0]
counter = np.zeros((13,))
for t in range(timesteps):
    reward = 0
    # request events will happen at random with a uniform prob of lambd[c]
    for class_num, prob in enumerate(lambd): # iterate through classes
        if random.random() < prob: # if there is a request
            action = Q.best_action(state, class_num)
            delta_state, taken = take_action(state) if action else (0, 0) 
            s_prime = state + delta_state # Take Action
            reward = 0 # set default reward to 0
            if taken:
                droptimes.append(t + np.random.exponential(beta)) # calculate droptime of resources allocated
                active_resources.append(delta_state)
                ###### Write rewards here, could be a function that takes in class_num and weather it is a fog node or not
                reward = rewards[class_num]
            # Store reward in array w.r.t the allocated number of requests
            if not (state==[0,0,0]).all():
                num_requests = int((state/delta_res).sum()/3)
                if num_requests > len(avg_rew_per_req)-1:
                    counter[num_requests] += 1
                    avg_rew_per_req.append(reward)
                else:
                    counter[num_requests] += 1
                    avg_rew_per_req[num_requests] = avg_rew_per_req[num_requests] + reward
            else:
                counter[0] += 1
            state = s_prime # set new stat

    i = 0 
    while i < len(active_resources):
        if t > droptimes[i]: # same as delta = 3
            state -= active_resources[i] # remove active resoueces from state
            droptimes.pop(i) 
            active_resources.pop(i)
        else:
            i += 1

for i, (avg_rew, cnt) in enumerate(zip(avg_rew_per_req, counter)):
    avg_rew_per_req[i] = avg_rew/cnt
    if i != 0:
        avg_reward.append(avg_rew/cnt+avg_reward[i-1])
    else:
        avg_reward = [0]

plt.plot(avg_reward[:-1], label="Q learning")
plt.plot(optimum[:len(avg_reward)-1,3], label="Optimum")
plt.title("Comparison of the optimal allocation of classes to the Q-learning")
plt.xlabel("Allocated requests",fontsize=13)
plt.ylabel("Average/optimum reward",fontsize=13)
plt.legend()
plt.show()