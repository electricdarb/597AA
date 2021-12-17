import random 
import numpy as np
import tensorflow as tf
from math import floor

from tensorflow.python.ops.gen_array_ops import one_hot

class DoubleDeep(tf.keras.Model):
    def __init__(self, max_resources, num_classes = 3):
        super(DoubleDeep, self).__init__()
        self.max_resources = tf.convert_to_tensor(max_resources, dtype = tf.float32)
        self.num_classes =  num_classes
        self.dense1 = tf.keras.layers.Dense(5, activation = tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(5, activation = tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(2, activation = tf.nn.softmax)

    def call(self, state, class_num):
        state = tf.divide(state, self.max_resources)
        one_hot_class = tf.one_hot(class_num, self.num_classes)
        x = tf.concat([state, one_hot_class], axis = 1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

class Qtable():
    def __init__(self, max_resources,
         num_classes = 3, 
         buffer_size = 4, 
         loss = tf.keras.losses.huber,
         opt = tf.keras.optimizers.SGD(),
         gamma = .97):
        """
        args:
            max_resources: list with each cell specifying resource capacity of each resourse
            blocks_per_resource: how many blocks to break resources up into, make resources discrete
            num_classes: number or classes or slices 
            actions: list of possible actions 
        self.table:  a q table that has an element for every reource, class, action combination
        """
        self.max_resources = max_resources
        self.Q_t = self.Q = DoubleDeep(max_resources, num_classes) # define target and regular q table
        num_resources = len(max_resources)
        self.buffer = tf.zeros((buffer_size, num_resources, 1, 1, num_resources))
        self.loss = loss
        self.opt = opt
        self.gamma = gamma
        
    def update_table(self, s, a, reward, class_num, s_prime):
        # bellman eq: Q(s, a) = Q(s, a) + alpha * (r + gamma * max(Q(s', a) - Q(s, a)))
        index = tf.math.argmax(self.Q_t(s_prime, class_num), axis = 1, output_type = tf.int32)
        hold_index = tf.concat([tf.reshape(tf.range(index.shape[0]), (-1, 1)), tf.reshape(index, (-1, 1))], axis = 1)  
        hold_a = tf.concat([tf.reshape(tf.range(index.shape[0]), (-1, 1)), tf.reshape(index, (-1, 1))], axis = 1)  
        with tf.GradientTape() as tape:
            y = reward + self.gamma * tf.gather_nd(self.Q(s_prime, class_num), hold_index)
            loss = self.loss(y, tf.gather_nd(self.Q(s, class_num), hold_a))
        grads = tape.gradient(loss, self.Q.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.Q.trainable_variables))

    def set_Qs_equal(self):
        self.Q_t.set_weights(self.Q.get_weights()) 

    def best_action(self, state, class_num):
        return int(tf.math.argmax(self.Q_t(state, [class_num]), axis = 1))

class Buffer():
    def __init__(self, max_size = 10000):
        self.max_size = max_size
        self.list = []

    def append(self, item):
        if len(self.list) >= self.max_size - 1:
            self.list.pop(random.randint(0, self.max_size - 2))
        self.list.append(item)

    def __len__(self):
        return len(self.list)


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
    state = np.zeros((1, num_resources)) # init state to 0 for all resources

    def take_action(state, num_class): # take action and prevent any invalid state from happened 
        delta_state = np.array([delta_comp, delta_radio, delta_storage])
        return delta_state
    
    def train(batch):
        state = tf.convert_to_tensor(np.concatenate(batch[:, 0]))
        action = tf.reshape(tf.convert_to_tensor(batch[:, 1], dtype = tf.uint8), (batch_size, 1))
        reward = tf.reshape(tf.convert_to_tensor(batch[:, 2], dtype = tf.float32), (batch_size, 1))
        class_num = tf.reshape(tf.convert_to_tensor(batch[:, 3], dtype = tf.uint8), (batch_size, ))
        s_prime = tf.convert_to_tensor(np.concatenate(batch[:, 4]))
        Q.update_table(state, action, reward, class_num, s_prime)

    #### init Q learning  Hyperparams ####
    gamma = .99 # doscount factor, range (0, 1), higher value -> future is valued higher 
    epsilon = .3 # greedy search factor, range(0, 1), higher value -> more random choices are made
    alpha = .001 # learning rate

    opt = tf.keras.optimizers.SGD(alpha)

    Q = Qtable(max_resources, num_classes = num_classes, opt = opt,  gamma = gamma) # init Q table with 5 state blocks for each resource

    #### init time values ####
    timesteps = int(1e5) # total timesteps to take
    delta_t = 1/40  # 40 steps per hour
    
    lambd = [12 * delta_t, 8 * delta_t, 10 * delta_t] # probabilites that an event happens in a time step
    beta  = 1 / (3 * delta_t) # define beta for expontial distrobution for release time

    active_resources = [] # queue to store active resources in so they can be removed from state when dropped
    droptimes = [] # queue to store droptimes in so the active resources can be removed from 

    reward_avg = .8
    class_rewards = ([], [], [])

    #### replay buffer values ####
    buffer = Buffer(max_size = 10000) # make buffer with max size 10000

    C = 1000
    C_counter = 0
    batch_size = 64

    for t in range(timesteps):
        reward = 0
        # request events will happen at random with a uniform prob of lambd[c]
        for class_num, prob in enumerate(lambd): # iterate through classes
            if random.random() < prob: # if there is a request
                action = Q.best_action(state, class_num) if random.random() > epsilon else random.choice(actions)
                delta_state = take_action(state, class_num) if action else 0
                s_prime = state + delta_state # Take Action
                resource_totals = np.sum(s_prime, axis = 0)
                taken = True # can this action be taken or not 
                if action:
                    for i in range(len(resource_totals)):
                        if resource_totals[i] > max_resources[i]: 
                            taken = False
                            s_prime = state # set s prime to state since it wont be changing 
                            break
                reward = 0 # set default reward to 0
                if taken:
                    droptimes.append(t + np.random.exponential(beta)) # calculate droptime of resources allocated
                    active_resources.append(delta_state)
                    ###### Write rewards here, could be a function that takes in class_num and weather it is a fog node or not
                    reward = rewards[class_num]
                buffer.append([state, action, reward, class_num, s_prime])
                state = s_prime # set new stat
                reward_avg = reward_avg * .999 + .001 * reward

                C_counter += 1
                if C_counter % C == 0: # 
                    Q.set_Qs_equal()
                    C_counter == 0
                
                if len(buffer) >=  batch_size:
                    batch = np.random.permutation(buffer.list)[:batch_size]
                    train(batch)

        i = 0 
        while i < len(active_resources):
            if t > droptimes[i]: # same as delta = 3
                state -= active_resources[i] # remove active resoueces from state
                droptimes.pop(i) 
                active_resources.pop(i)
            else:
                i += 1 
        
        if t % 1000 == 0:
            print(reward_avg, ' ', epsilon)
            #epsilon *= .9975