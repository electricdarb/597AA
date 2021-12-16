import tensorflow as tf
import tensorflow.keras as keras
from copy import deepcopy
import numpy as np

from DDnetwork import DeepNN
from Qtable import Qtable, next_state

# Initialize replay memory to capacity D
replay_memory = []
# Initialize the Q-network with random weights
q_net = DeepNN()
# Initialize the target Q-network with same weights -> that's why deepcopy
target_net = deepcopy(q_net)

T = 100
epsilon = 0.01
gamma = 0.1
state = [0,0,0]
reward = 0
C = 10
optimizer = keras.optimizers.SGD(learning_rate=1e-3)
for episode in range(T):
    if tf.random.uniform(0, minval=0, maxval=1) > epsilon:
        action = tf.random.uniform(0, minval=0, maxval=1, dtype=tf.dtypes.int8)
    else:
        action = Qtable.best_action(state, action)
        slice_ = tf.random.uniform(shape=[], minval=0, maxval=3, dtype=tf.int64)
        state_ = next_state(state, slice_, action)
        replay_memory.append([[state, action, reward, state_]])
        s_j, a_sj, r_j, s_j1 = replay_memory[tf.random.uniform(shape=(),minval=0,maxval=len(replay_memory),dtype=tf.int32)]
        action_hat = tf.argmax(target_net(s_j,a_sj,s_j1,r_j))
        with tf.GradientTape() as tape:
            q_value = q_net(s_j,action_hat,s_j1,r_j)[action_hat]
            y_j = r_j + gamma*q_value
            loss = (y_j-q_value)**2
        grads = tape.gradient(loss, q_net.trainable_weights)
        optimizer.apply_gradients(zip(grads, q_net.trainable_weights))
        if episode%C == C-1:
            target_net = deepcopy(q_net)



q_net.compile(optimizer='adam',
              loss=keras.losses.MeanSquaredError(),
              metrics=['accuracy'])
# Fit model
q_net.fit(np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]), np.array([[0,0],[0,0],[0,0],[0,0]]), epochs = 10)

# q_net.summary()
# input_t = tf.constant([1,2,3,4], dtype=tf.float32)


