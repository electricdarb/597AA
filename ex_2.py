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

# avialable fog resources
radio_fog = None
comp_fog = None
storage_fog = None 

# avialable InP resources
radio_inp = None
comp_inp = None
storage_inp = None

# avialable resources 
radio_total = radio_fog + radio_inp
comp_total = comp_fog + comp_inp
storage_total = storage_fog + storage_inp 

# each request requires 50 Mbps for radio access, 2 CPUs for computing and 2 GB pf storage resources
radio_min = 100 # mbps
comp_min = 2 # cpus 
storage_min = 1 # GBs

# init how much of each resourse is used
radio_used = 0
comp_used = 0
storage_used = 0

all_situations = None # an exhaustive combo of 

# run every possibility of giving every slice different combos of resources and test reward

    
        
    