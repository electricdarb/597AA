import tensorflow as tf

class Qtable():
    def __init__(self, state_space_shape, num_actions):
        self.table = tf.zeros((*state_space_shape, num_actions)) #init Qtable
        self.state = tf.convert_to_tensor()
    def update_table(self, s, a, gamma, alpha):
        index_s = self.map_s(s)
        index_a = self.map_a(a)

        s_prime = self.next_state(s, a)
        index_s_prime = self.map_s(s_prime)
        
        self.table[index_s, index_a] = self.table[index_s, index_a] + alpha * (self.reward(s, a)
            + gamma * tf.math.reduce_max(self.table[index_s_prime, :]) - self.table[index_s, index_a])

    @staticmethod
    def map_s(s):
        pass
    
    @staticmethod
    def map_a(a):
        pass
    
    @staticmethod
    def reward(s, a):
        pass