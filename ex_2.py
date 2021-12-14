import numpy as np

## -------------------------------------------------------
## Parameter settings
## -------------------------------------------------------
''' Parameters not needed rn
# Number of slices = 3 (class-1, class-2, class-3)
num_slices = 3
# arrival rates for class-1, class-2, class-3
arr_1 = 12
arr_2 = 8
arr_3 = 10
# completion rates for class-1, class-2, class-3
compl_rate = 3
# immediate reward from class-1, class-2, class-3
reward_1 = 1
reward_2 = 2
reward_3 = 4
# required resources for slice request
storage_req = 1000
comp_req = 2
radio_req = 100
# maximum radio resources set to 500 Mbps
radio_max = 500
'''

# each request requires 50 Mbps for radio access, 2 CPUs for computing and 2 GB pf storage resources
request_resources = np.array([50, 2, 2]) # Mbps, CPUs, Gb

def greedy(arr_class_rew, rew, res_used, max_res):
    new_res = res_used + request_resources
    if (new_res<=max_res).all():
        res_used = new_res
        rew += arr_class_rew
        # if (new_res<=max_res-4*request_resources).all():
        #     if arr_class_rew == 4: # if resoruces are too high, wait for class-3
        #         res_used = new_res
        #         rew += arr_class_rew
        # else:
    return rew, res_used

def other(arr_class_rew, rew, res_used, max_res):
    new_res = res_used + request_resources
    if (new_res<=max_res).all():
        if (new_res<=max_res-request_resources).all():
            if arr_class_rew == 4: # if resoruces are too high, wait for class-3
                res_used = new_res
                rew += arr_class_rew
        else:
            res_used = new_res
            rew += arr_class_rew
    # else:
    #     if arr_class_rew == 4:
    #         print("Worst case!!")
    return rew, res_used

def completion(res_used):
    res_used -= request_resources*3 # *3 bc done for each slice
    return res_used

# small-size system
resources_small = np.array([400, 8, 4]) # Mbps, CPUs, Gb

fog_nodes = range(1,10)
algorithm = other
rewards = []
nodes = []
for num_nodes in fog_nodes:
    resources_used = np.array([0, 0, 0])
    max_resources = num_nodes*resources_small
    # Simulate requests per hour -> choose 120 so that it's dividable by 8
    reward = 0
    num_hours = 5
    for i in range(num_hours*120):
        if i%10==9:
            reward, resources_used = algorithm(1, reward, resources_used, max_resources)
        if i%15==14:
            reward, resources_used = algorithm(2, reward, resources_used, max_resources)
        if i%12==11:
            reward, resources_used = algorithm(4, reward, resources_used, max_resources)
        if i%40==39:
            resources_used = completion(resources_used)
    rewards.append(reward)
    nodes.append(num_nodes)
    print("Number of fog nodes ", num_nodes, ", reward ", reward)