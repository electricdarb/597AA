import tensorflow as tf
import random 

class Qtable():
    def __init__(self, num_classes: int, num_resources: int, actions: list = [0, 1]):
        """
        num_classes: int, number of classes in the model
        num_resources: int, maximum amount of resources that can be assigned for a given class
        """
        """
        request space: [[0, 1] for i in range(num_classes)]
        """
        self.actions = actions # accept or deny request
        num_actions = len(self.actions)
        # Q table, initialized to 0. 
        self.table = tf.zeros((num_classes, num_resources, num_actions)) # shape num_classes* num_resources
       
    def update_table(self, s: list, a: int, gamma, alpha):
        index_s = self.map_s(s)
        index_a = a 
        s_prime = self.next_state(s, a)
        index_s_prime = self.map_s(s_prime)
        
        self.table[index_s, index_a] = self.table[index_s, index_a] + alpha * (self.reward(s, a)
            + gamma * tf.math.reduce_max(self.table[index_s_prime, :]) - self.table[index_s, index_a])

    def best_action(self, state, slice_):
        return tf.argmax(self.table[state, :])

    @staticmethod
    def map_s(s):
        pass
    
    @staticmethod
    def map_a(a):
        pass
    
    @staticmethod
    def reward(s, a):
        pass


def next_state(state, slice_, action):
    if action == 1: # if resource is being taken
        pass
    elif action == -1: # if resource is being releases TODO: isn't it action = [0, 1] not [-1, 1]
        pass

if __name__ == "__main__":

    num_events  = 1e6 # define the number of iterations/events
    num_classes = 3 # there are three classes: utilities, automotive, and manufactuting 
    num_resources = 3 # there are three resources: radio, storage, and compute 

    gamma = 1
    epsilon = 1
    alpha = 1

    # action space
    actions = [0, 1]

    # init table with 3 classes and 3 resources
    Q = Qtable(num_classes, num_resources, actions)

    # init state
    state = None

    # init events, each event is shape (num_classes, 3)
    events = tf.zeros((num_events, 3)) # there is a more effiecent way of doing this, space wise but this is what the paper does
    delta_t = 
    for i in range(num_events):
        pass # loop over and define each event 

    for event in events:
        for slice_, request in enumerate(event):
            if request  == 1: # if a request is being made 
                action = Q.best_action(state, slice_) if random.random() > epsilon else random.choice(actions)
                Q.update_table(state, slice_, action)
                state = next_state(state, slice_, 1) if action else state # take action if action 
                break # break bc paper says each event can only correspond to one class
            elif request == -1: # if a release is being made
                state = next_state(state, slice_, -1) # release resource 
                break # break bc paper says each event can only correspond to one class