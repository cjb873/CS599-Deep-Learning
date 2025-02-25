"""
author:-aam35
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
learning_rate = None
batch_size = None
n_epochs = None
n_train = None
n_test = None

# Step 1: Read in data
from tensorflow.keras.datasets import fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0  # Normalize data


val_split = 0.1
num_val = int(train_images.shape[0] * val_split)
val_images = train_images[:num_val]
val_labels = train_labels[:num_val]
train_images_split = train_images[num_val:]
train_labels_split = train_labels[num_val:]


# create one iterator and initialize it with different datasets
iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                           train_data.output_shapes)
img, label = iterator.get_next()

train_init = iterator.make_initializer(train_data)	# initializer for train_data
test_init = iterator.make_initializer(test_data)	# initializer for train_data

# Step 3: create weights and bias
# w is initialized to random variables with mean of 0, stddev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# shape of b depends on Y
w, b = None, None
#############################
########## TO DO ############
#############################


# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
logits = None
#############################
########## TO DO ############
#############################


# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
loss = None
#############################
########## TO DO ############
#############################


# Step 6: define optimizer
# using Adam Optimizer with pre-defined learning rate to minimize loss
optimizer = None
#############################
########## TO DO ############
#############################


# Step 7: calculate accuracy with test set
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

#Step 8: train the model for n_epochs times
for i in range(n_epochs):
	total_loss = 0
	n_batches = 0
	#Optimize the loss function
	print("Train and Validation accuracy")
	################################
	###TO DO#####
	############

#Step 9: Get the Final test accuracy

#Step 10: Helper function to plot images in 3*3 grid
#You can change the function based on your input pipeline

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
images = test_data[0:9]

# Get the true classes for those images.
y = test_class[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images=images, y=y)


#Second plot weights

def plot_weights(w=None):
    # Get the values for the weights from the TensorFlow variable.
    #TO DO ####

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = None
    #TO DO## obtains these value from W
    w_max = None

    # Create figure with 3x4 sub-plots,
    # where the last 2 sub-plots are unused.
    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Only use the weights for the first 10 sub-plots.
        if i<10:
            # Get the weights for the i'th digit and reshape it.
            # Note that w.shape == (img_size_flat, 10)
            image = w[:, i].reshape(img_shape)

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

