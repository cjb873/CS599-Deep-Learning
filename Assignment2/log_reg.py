"""
author:-cjb873
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import utils

# Define paramaters for the model
n_train = 2000
n_test = 200

# Step 1: Read in data
from tensorflow.keras.datasets import fashion_mnist

def get_data(val_split=0.1):
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0  # Normalize data

    num_val = int(train_images.shape[0] * val_split)
    val_images = train_images[:num_val]
    val_labels = train_labels[:num_val]
    train_images_split = train_images[num_val:]
    train_labels_split = train_labels[num_val:]
    return train_images_split, train_labels_split, val_images, val_labels, test_images, test_labels


train_images_split, train_labels_split, val_images, val_labels, test_images, test_labels = get_data()

train_data, test_data = train_images_split, test_images

# Step 3: create weights and bias
# w is initialized to random variables with mean of 0, stddev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# shape of b depends on Y
w, b = None, None
#############################
########## TO DO ############
#############################
w = tf.Variable(tf.random.normal([28*28, 10], stddev=.01), name='weights')
b = tf.Variable(tf.zeros([10]), name='biases')

# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
logits = None
#############################
########## TO DO ############
#############################
class LogisticRegressionModel(tf.Module):
    def __init__(self):
        super().__init__()
        self.W = tf.Variable(tf.random.normal([28*28, 10], stddev=0.01), name='weights')
        self.b = tf.Variable(tf.zeros([10]), name="biases")

    def __call__(self, x):
        x = tf.cast(x, tf.float32)
        x = tf.reshape(x, [-1, 28*28])
        logits = tf.matmul(x, self.W) + self.b
        return tf.nn.softmax(logits)



# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
loss = None
#############################
########## TO DO ############
#############################
def compute_loss(model, images, labels, lambda_reg=0.0):
    predictions = model(images)
    ce_loss = tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(labels, predictions))
    l2_loss = tf.nn.l2_loss(model.W)  # Sum of squares divided by 2
    return ce_loss + lambda_reg * l2_loss



# Step 6: define optimizer
# using Adam Optimizer with pre-defined learning rate to minimize loss
optimizer = None
#############################
########## TO DO ############
#############################
optimizer = tf.optimizers.Adam(learning_rate=learning_rate)



# Step 7: calculate accuracy with test set
model = LogisticRegressionModel()
preds = model(test_images)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(test_labels, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

#Step 8: train the model for n_epochs times
def train_step(model, images, labels, optimizer, lambda_reg=0.0):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, images, labels, lambda_reg)
    # Compute gradients with respect to the model's variables.
    grads = tape.gradient(loss, [model.W, model.b])
    # Update the variables using the optimizer.
    optimizer.apply_gradients(zip(grads, [model.W, model.b]))
    return loss

def train_model(optimizer, lambda_reg=0.0, num_epochs=20, batch_size=128, split=0.1):
    tf.random.set_seed(99111100121)
    # Create a new model instance (fresh initialization)
    model = LogisticRegressionModel()
    # Prepare tf.data datasets for training and validation
    train_images_split, train_labels_split, val_images, val_labels, test_images, test_labels = get_data(split)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images_split, train_labels_split))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    val_dataset = val_dataset.batch(batch_size)
    # Lists to record metrics for each epoch.
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }
    # For accuracy computation
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    for epoch in range(0, num_epochs):
        # Reset metrics at the start of each epoch.
        train_acc_metric.reset_state()
        val_acc_metric.reset_state()

        # Training loop
        epoch_losses = []
        for batch_images, batch_labels in train_dataset:
            loss = train_step(model, batch_images, batch_labels, optimizer, lambda_reg)
            epoch_losses.append(loss.numpy())
            # Update training accuracy.
            predictions = model(batch_images)
            train_acc_metric.update_state(batch_labels, predictions)

        # Compute average training loss over epoch.
        train_loss = np.mean(epoch_losses)
        train_accuracy = train_acc_metric.result().numpy()

        # Validation loop
        val_losses = []
        for batch_images, batch_labels in val_dataset:
            loss = compute_loss(model, batch_images, batch_labels, lambda_reg)
            val_losses.append(loss.numpy())
            predictions = model(batch_images)
            val_acc_metric.update_state(batch_labels, predictions)
        val_loss = np.mean(val_losses)
        val_accuracy = val_acc_metric.result().numpy()

        # Save metrics
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        print(f"Epoch {epoch:02d}: "
              f"Train Loss = {train_loss:.4f}, Train Acc = {train_accuracy:.4f}, "
              f"Val Loss = {val_loss:.4f}, Val Acc = {val_accuracy:.4f}")
    return model, history

learning_rate = 0.001
batch_size = 128
n_epochs = 20
lam = 0.00
model, history = train_model(optimizer, lambda_reg=lam, num_epochs=n_epochs, batch_size=batch_size)

#Step 9: Get the Final test accuracy

#Step 10: Helper function to plot images in 3*3 grid
#You can change the function based on your input pipeline
img_shape = [28, 28]
def plot_images(images, y, yhat=None):
    assert len(images) == len(y) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if yhat is None:
            xlabel = "True: {0}".format(y[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(y[i], yhat[i])

        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

#Get image from test set
images = test_data[20:29]
# Get the true classes for those images.
y = test_labels[20:29]
yhat = tf.argmax(model(images), 1)

# Plot the images and labels using our helper-function above.
plot_images(images=images, y=y, yhat=yhat)


#Second plot weights

def plot_weights(w=model.W):
    # Get the values for the weights from the TensorFlow variable.
    #TO DO ####

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = tf.reduce_min(w)
    #TO DO## obtains these value from W
    w_max = tf.reduce_min(w)
    print(f"Min: {w_min}, Max: {w_max}")
    # Create figure with 3x4 sub-plots,
    # where the last 2 sub-plots are unused.
    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Only use the weights for the first 10 sub-plots.
        if i<10:
            # Get the weights for the i'th digit and reshape it.
            # Note that w.shape == (img_size_flat, 10)
            print(w[:,i].shape)
            image = tf.reshape(w[:, i], img_shape)
            print(image)
            # Set the label for the sub-plot.
            ax.set_xlabel("Weights: {0}".format(i))

            # Plot the image.
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

        # Remove ticks from each sub-plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()



optimizers_to_try = {
    "SGD": tf.optimizers.SGD(learning_rate=0.5),
    "Adam": tf.optimizers.Adam(learning_rate=0.001),
    "RMSprop": tf.optimizers.RMSprop(learning_rate=0.001)
}
epochs_to_try = {
    "1": 1,
    "10": 10,
#    "100": 100
}

bs_to_try = {
    "16": 16,
    "64": 64,
    "128": 128,
    "512": 512
}

splits_to_try = {
    "0": 0.,
    "0.1": 0.1,
    "0.2": 0.2,
    "0.5": 0.5
}

lambda_reg = 0.10  # Change to 0.0 to see training without regularization.
num_epochs = 10  # Adjust as needed

def train_and_plot(in_dict, hyperparam):
    opt = tf.optimizers.Adam(learning_rate=0.001)
    num_epochs = epochs_to_try["10"]
    bs = bs_to_try["128"]
    split = splits_to_try["0.1"]

    history_dict = {}
    for name, item in in_dict.items():

        if hyperparam == "opt":
            opt = item
        elif hyperparam == "epochs":
            num_epochs = item
        elif hyperparam == "bs":
            bs = item
        elif hyperparam == "split":
            split = item

        print(f"Training with {name} (lambda_reg={lambda_reg})")
        # Train a new model for each optimizer.
        history = train_model(opt, lambda_reg=lambda_reg, num_epochs=num_epochs, batch_size=bs, split=split)[1]
        history_dict[name] = history
    epochs = range(1, num_epochs + 1)
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    for name, history in history_dict.items():
        axs[0].plot(epochs, history["train_loss"], label=f"{name} Train")
        axs[0].plot(epochs, history["val_loss"], '--', label=f"{name} Val")
    axs[0].set_title("Loss over Epochs")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    for name, history in history_dict.items():
        axs[1].plot(epochs, history["train_accuracy"], label=f"{name} Train")
        axs[1].plot(epochs, history["val_accuracy"], '--', label=f"{name} Val")
    axs[1].set_title("Accuracy over Epochs")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


train_and_plot(optimizers_to_try, "opt")
train_and_plot(epochs_to_try, "epochs")
train_and_plot(bs_to_try, "bs")
train_and_plot(splits_to_try, "split")
