import numpy as np
import matplotlib.pyplot as plt
from math import ceil
## -------------------------------------------------------
## Parameter settings
## -------------------------------------------------------

# immediate reward from class-1, class-2, class-3
rewards = (1, 2, 4)

delta_comp = 2 # CPU
delta_storage = 1 # GB
delta_radio = 100 # Mbps

delta_res = np.array([delta_comp, delta_radio, delta_storage])

results = []

# run over different number of fog nodes
avail_res = [5, 5, 5]  
max_fog_nodes = np.array(avail_res).sum()
resource_splits = []
total_rewards = []
max_rew = np.zeros((avail_res[0]+avail_res[1]+avail_res[2]-2, 4))
state = np.array([0, 0, 0])
# loop over every combination of resources
for slice_1 in range(avail_res[0]):
    for slice_2 in range(avail_res[1]):
        for slice_3 in range(avail_res[2]):
            state = delta_res * (slice_1+slice_2+slice_3)
            reward = 0
            allocated_fog_nodes = slice_1+slice_2+slice_3
            if allocated_fog_nodes > max_fog_nodes:
                # way to first allocate eMMB slices, then mMTC, and then URLLC
                available_nodes = max_fog_nodes

                if3 = slice_3 <= available_nodes
                reward += slice_3 * rewards[2] if if3 else available_nodes * rewards[2]
                available_nodes = available_nodes-slice_3 if if3 else 0

                if2 = slice_2 <= available_nodes
                reward += slice_2 * rewards[1] if if2 else available_nodes * rewards[1]
                available_nodes = available_nodes-slice_2 if if2 else 0

                if1 = slice_1 <= available_nodes
                reward += slice_1*rewards[0] if if1 else available_nodes * rewards[0] # this should always be else
                available_nodes = available_nodes-slice_1 if if1 else 0

            else:
                reward = slice_1 * rewards[0] + slice_2 * rewards[1] + slice_3 * rewards[2]
            if reward >= max_rew[allocated_fog_nodes, 3]:
                max_rew[allocated_fog_nodes] = [slice_1, slice_2, slice_3, reward]

            resource_splits.append(avail_res)
            total_rewards.append(reward)

max_reward = max(total_rewards)
max_idx = total_rewards.index(max_reward)
results.append({'maximum resources' : resource_splits[max_idx],
                'maximum rewards' : max_reward })

np.save("optimum.npy", max_rew)