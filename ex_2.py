import numpy as np

## -------------------------------------------------------
## Parameter settings
## -------------------------------------------------------

# arrival rates for class-1, class-2, class-3
arr_1 = 12
arr_2 = 8
arr_3 = 10

# completion rates for class-1, class-2, class-3
compl_rate = 3

# immediate reward from class-1, class-2, class-3
reward_1 = 1
reward_2 = 2
reward_3 = 3

# required resources for slice request
storage_req = 1000
comp_req = 2
radio_req = 100

# maximum radio resources set to 500 Mbps
radio_max = 500

# each request requires 50 Mbps for radio access, 2 CPUs for computing and 2 GB pf storage resources
radio_min = 50
comp_min = 2
storage_min = 2

for comp in range(1,10):
    for storage in range(1,10):
        if comp >= comp_min:
            if storage >= storage_min:
                print("Computing and storage sufficient")
        pass