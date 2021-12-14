import tensorflow as tf

class DeepNN(tf.keras.Model):
    def __init__(self):
        super(DeepNN, self).__init__()
        # self.input = tf.keras.Input(shape=(4,))
        # self.inp = tf.keras.layers.Input(shape=(4,1))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(5, activation= tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(5, activation= tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(2, activation= tf.nn.relu)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x