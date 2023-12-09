# =============================================================================
# PROBLEM B2
#
# Build a classifier for the Fashion MNIST dataset.
# The test will expect it to classify 10 classes.
# The input shape should be 28x28 monochrome. Do not resize the data.
# Your input layer should accept (28, 28) as the input shape.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 83%
# =============================================================================

import tensorflow as tf


def solution_B2():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    class Callback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs.get('accuracy') is not None and logs.get('accuracy') > 0.83:
                print("\nReached 83% accuracy so cancelling training!")
                self.model.stop_training = True

    # NORMALIZE YOUR IMAGE HERE
    (x_train, y_train), _ = fashion_mnist.load_data()
    x_train = x_train / 255.0

    # DEFINE YOUR MODEL HERE
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        # End with 10 Neuron Dense, activated by softmax
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    # COMPILE MODEL HERE
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # TRAIN YOUR MODEL HERE
    history = model.fit(x_train, y_train, epochs=10, callbacks=[Callback()])

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B2()
    model.save("model_B2.h5")
