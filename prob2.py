import numpy as np
import matplotlib.pyplot as plt
## -------------------------------------------------------
## Parameter settings
## -------------------------------------------------------

# arrival rates for class-1, class-2, class-3 per hour
arr_1 = 12 
arr_2 = 8
arr_3 = 10

# completion rates for class-1, class-2, class-3 per hour
compl_rate = 3 

# immediate reward from class-1, class-2, class-3
rewards = (1, 2, 4) 

# slice requests per hour from each class 
request_rate = (12, 8, 10)

# resources per fog node 
node_res = 1

# avialable InP resources
inp_res = 4 

# each request requires 50 Mbps for radio access, 2 CPUs for computing and 2 GB pf storage resources
radio_min = 100 # mbps
comp_min = 2 # CPUs
storage_min = 1 # GBs

# init how much of each resourse is used
radio_used = 0
comp_used = 0
storage_used = 0

max_fog_nodes = 10

results = []

# run over different number of fog nodes , start at 10 to make sense
for num_nodes in range(10):
    # avialable resources for each slice, (inp_res) + (chosen_fog_nodes) * (resources_per_node)
    avial_res = [inp_res+4*num_nodes*node_res, inp_res+2*num_nodes*node_res, inp_res+num_nodes*node_res]  

    resource_splits = []
    total_rewards = []
    # loop over every combination of resources
    for slice_1 in range(avial_res[0]):
        for slice_2 in range(avial_res[1]):
            for slice_3 in range(avial_res[2]):
                reward = 0
                allocated_fog_nodes = slice_1+slice_2+slice_3
                if allocated_fog_nodes > max_fog_nodes:
                    # way to first allocate eMMB slices, then mMTC, and then URLLC
                    available_nodes = max_fog_nodes

                    if1 = slice_1-inp_res <= available_nodes
                    reward += slice_1*rewards[0] if if1 else available_nodes * rewards[0]
                    available_nodes = available_nodes-slice_1 if if1 else 0

                    if2 = slice_2-inp_res <= available_nodes
                    reward += slice_2 * rewards[1] if if2 else available_nodes * rewards[1]
                    available_nodes = available_nodes-slice_2 if if2 else 0

                    if3 = slice_3-inp_res <= available_nodes
                    reward += slice_3 * rewards[2] if if3 else available_nodes * rewards[2] # this should always be else
                    available_nodes = available_nodes-slice_3 if if3 else 0

                else:
                    reward = slice_1 * rewards[0] + slice_2 * rewards[1] + slice_3 * rewards[2]

                resource_splits.append(avial_res)
                total_rewards.append(reward)

    max_reward = max(total_rewards)
    max_idx = total_rewards.index(max_reward)
    results.append({'maximum resources' : resource_splits[max_idx],
                    'maximum rewards' : max_reward })

plt.plot([res['maximum rewards'] for res in results])
plt.xlabel('Number of fog nodes')
plt.ylabel('Maximum reward')
plt.show()
print('Done')