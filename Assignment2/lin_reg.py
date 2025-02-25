"""
author:-cjb873
"""
import time

import tensorflow as tf
import matplotlib.pyplot as plt


tf.random.set_seed(99111100121)
# Create data
NUM_EXAMPLES = 100000

# define inputs and outputs with some noise
X = tf.random.normal([NUM_EXAMPLES])  # inputs

noise_level = 1.
noise_y = tf.random.normal([NUM_EXAMPLES], stddev=noise_level) # noise
#noise = tf.random.uniform([NUM_EXAMPLES], minval=-noise_level, maxval=noise_level)
#noise = np.random.laplace(loc=0., scale=noise_level, size=[NUM_EXAMPLES])

noise_x = tf.random.normal([NUM_EXAMPLES], stddev=noise_level)



y = X * 3 + 2 + noise_y  # true output

# Create variables.
W = tf.Variable(0.0)
b = tf.Variable(10000.0)


train_steps = 1000
learning_rate = 0.01


def prediction(x):
    # noise_w = tf.random.normal([1], stddev=noise_level)
    return (x+noise_x) * W + b


def MSE_loss(y, yhat):
    return tf.reduce_mean(tf.square(yhat - y))


def MAE_loss(y, yhat):
    return tf.reduce_mean(tf.abs(yhat - y))


def hybrid_loss(y, yhat, alpha=0.25):
    return tf.reduce_mean(alpha * tf.abs(yhat - y) +
                          (1. - alpha) * tf.square(yhat - y))


loss_name = "MSE"
losses = {"MSE": MSE_loss, "MAE": MAE_loss, "hybrid": hybrid_loss}

patience = 300            # Number of steps to wait before reducing LR
lr_decay_factor = 0.5     # Factor to reduce LR by (e.g., multiply by 0.5)
best_loss = float('inf')  # Initialize best loss as infinity
patience_counter = 0      # Counter to track how long loss has not improved


for i in range(train_steps):

    with tf.GradientTape() as tape:

        yhat = prediction(X)
        loss = losses[loss_name](yhat, y)

    dW, db = tape.gradient(loss, [W, b])

    #lr_noise = tf.random.normal([], stddev=0)
    W.assign_sub((learning_rate) * dW)
    b.assign_sub((learning_rate) * db)

    current_loss = loss.numpy()

    if current_loss < best_loss:
        best_loss = current_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        learning_rate *= lr_decay_factor
        print(f"Reducing learning rate to {learning_rate:.6f} at step {i}")
        patience_counter = 0  # Reset the counter after reducing LR

    # Print training progress every 500 steps
    if i % 500 == 0:
        print(f"Step {i}, Loss: {current_loss:.4f}, W: {W.numpy():.4f}, b: {b.numpy():.4f}")

print(f"\nFinal Model: W = {W.numpy():.4f}, b = {b.numpy():.4f}, Final Loss: {loss.numpy():.4f}")
