import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.layers import RNN, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.preprocessing import image_dataset_from_directory
print("TensorFlow version:", tf.__version__)


assert tf.__version__.startswith("2"), "Please use TensorFlow 2.x"

image_dir = "notMNIST_small"
cm = "grayscale"
seed = 0
split = 0.2
train_ds, test_ds = image_dataset_from_directory(image_dir,
                                                 color_mode=cm,
                                                 image_size=(28, 28),
                                                 seed=seed,
                                                 validation_split=split,
                                                 subset="both",
                                                 label_mode='categorical'
                                                 )
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))


class BaseRNNCell(Layer):
    def __init__(self, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size

    @property
    def state_size(self):
        # Keras will allocate a state tensor of shape (batch, hidden_size)
        return self.hidden_size

    def build(self, input_shape):
        # input_shape = (batch, feature_dim)
        # We don't create weights here—we let subclasses do that.
        super().build(input_shape)

    def call(self, inputs, states):
        # inputs: (batch, feature_dim)
        raise NotImplementedError("Must implement in subclass")


class GRUCell(BaseRNNCell):
    def build(self, input_shape):
        # input_shape[-1] = D
        D = input_shape[-1]
        H = self.hidden_size
        total_dim = D + H  # we'll concatenate [h_prev, x]

        # 1) Update gate parameters: zₜ = sigmoid([h_prev, x]·Wz + bz)
        self.Wz = self.add_weight(
            name="Wz", shape=[total_dim, H], initializer="glorot_uniform")
        self.bz = self.add_weight(
            name="bz", shape=[H], initializer="zeros")

        # 2) Reset gate parameters: rₜ = sigmoid([h_prev, x]·Wr + br)
        self.Wr = self.add_weight(
            name="Wr", shape=[total_dim, H], initializer="glorot_uniform")
        self.br = self.add_weight(
            name="br", shape=[H], initializer="zeros")

        # 3) Candidate: ŧhₜ = tanh([r⊙h_prev, x]·Ws + bs)
        self.Ws = self.add_weight(
            name="Ws", shape=[total_dim, H], initializer="glorot_uniform")
        self.bs = self.add_weight(
            name="bs", shape=[H], initializer="zeros")

        super().build(input_shape)  # finalize build

    def call(self, inputs, states):
        # inputs: (batch, D), states[0]=h_prev: (batch, H)
        h_prev = states[0]

        # 1) Concatenate previous state & input
        concat_hx = tf.concat([h_prev, inputs], axis=-1)  # → (batch, H+D)

        # 2) Compute update & reset gates
        z = tf.sigmoid(tf.matmul(concat_hx, self.Wz) + self.bz)
        r = tf.sigmoid(tf.matmul(concat_hx, self.Wr) + self.br)

        # 3) Compute candidate hidden state
        gated_h = r * h_prev                             # (batch, H)
        concat_candidate = tf.concat([gated_h, inputs], axis=-1)
        h_tilde = tf.tanh(tf.matmul(concat_candidate, self.Ws) + self.bs)

        # 4) Final new hidden state
        new_h = (1 - z) * h_prev + z * h_tilde            # (batch, H)

        # Return output (we choose to output the hidden state) and new state
        return new_h, [new_h]


class MGUCell(BaseRNNCell):
    def build(self, input_shape):
        D = input_shape[-1]
        H = self.hidden_size
        total_dim = D + H

        # 1) Single gate fₜ
        self.Wf = self.add_weight(
            name="Wf", shape=[total_dim, H], initializer="glorot_uniform")
        self.bf = self.add_weight(
            name="bf", shape=[H], initializer="zeros")

        # 2) Candidate hidden state (same as GRU’s Ws, bs)
        self.Ws = self.add_weight(
            name="Ws", shape=[total_dim, H], initializer="glorot_uniform")
        self.bs = self.add_weight(
            name="bs", shape=[H], initializer="zeros")

        super().build(input_shape)

    def call(self, inputs, states):
        h_prev = states[0]

        # 1) Concatenate [h_prev; x]
        concat_hx = tf.concat([h_prev, inputs], axis=-1)

        # 2) Compute gate fₜ
        f = tf.sigmoid(tf.matmul(concat_hx, self.Wf) + self.bf)

        # 3) Candidate uses f as reset
        gated_h = f * h_prev
        concat_candidate = tf.concat([gated_h, inputs], axis=-1)
        h_tilde = tf.tanh(tf.matmul(concat_candidate, self.Ws) + self.bs)

        # 4) New state
        new_h = (1 - f) * h_prev + f * h_tilde

        return new_h, [new_h]


def build_rnn_model(cell_cls, hidden_size, num_layers, num_classes=10):
    # 1. Instantiate each layer’s cell
    cells = [cell_cls(hidden_size) for _ in range(num_layers)]
    # 2. Wrap in a single RNN layer
    rnn_stack = RNN(cells, return_sequences=False)  # only final output

    # 3. Define input placeholder: (time_steps=28, features=28, channels=1)
    inp = Input(shape=(28, 28, 1))
    # 4. Drop the channel dimension → shape (28,28)
    x = tf.squeeze(inp, axis=-1)

    # 5. Run through RNN stack → final hidden state (batch, H)
    h_final = rnn_stack(x)

    # 6. Project to 10 classes
    logits = Dense(num_classes)(h_final)
    return Model(inputs=inp, outputs=logits)


def compile_and_train(model, train_ds, test_ds, epochs=20):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    # Fit returns a History object with loss/accuracy per epoch
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs
    )
    return history


def set_seed(seed=0):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


results = {}  # store histories

for cell_name, cell_cls in [("GRU", GRUCell), ("MGU", MGUCell)]:
    for num_layers in [1, 2, 3, 4]:
        for hidden_size in [50, 128]:
            config_key = f"{cell_name}_{num_layers}l_{hidden_size}h"
            results[config_key] = []
            for trial in range(3):
                print(f"Training w/: {config_key}")
                seed = 1000 + trial
                set_seed(seed)

                # Build & train
                model = build_rnn_model(cell_cls, hidden_size, num_layers)
                history = compile_and_train(model, train_ds, test_ds,
                                            epochs=20)

                # Save the per-epoch metrics
                results[config_key].append(history.history)


def plot_metric(results, metric="accuracy", val=True):
    plt.figure(figsize=(8, 5))
    for key, runs in results.items():
        # runs: list of 3 history dicts
        vals = np.array([r[f"val_{metric}"] for r in runs])
        mean = vals.mean(axis=0)
        std = vals.std(axis=0)
        epochs = np.arange(1, len(mean)+1)

        # Shade std region
        plt.fill_between(epochs, mean-std, mean+std, alpha=0.2)
        # Plot mean curve
        plt.plot(epochs, mean, label=key)

    plt.xlabel("Epoch")
    plt.ylabel(f"Validation {metric.capitalize()}")
    plt.legend(fontsize="small", ncol=2)
    plt.title(f"Mean ± Std of val_{metric} over Trials")
    plt.show()


# Example: plot validation accuracy
plot_metric(results, metric="accuracy")
