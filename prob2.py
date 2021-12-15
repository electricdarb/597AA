import numpy as np

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

max_fog_nodes = 50

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
                if slice_1+slice_2+slice_3-3*inp_res > max_fog_nodes:
                    # find a way to first allocate eMMB slices, then mMTC, and then URLLC
                    reward = 0
                    pass
                else:
                    reward = slice_1 * rewards[0] + slice_2 * rewards[1] + slice_3 * rewards[2]
                resource_splits.append(avial_res)
                total_rewards.append(reward)
    results.append({'resource_splits' : resource_splits,
                    'rewards' : total_rewards })