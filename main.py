import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop

from random import seed
from random import random

from tensorflow import keras

import sys

# class CustomModel(keras.models.Sequential):
#     def train_step(self, data):
#         # Unpack the data. Its structure depends on your model and
#         # on what you pass to `fit()`.
#         x, y = data
#
#         with tf.GradientTape() as tape:
#             y_pred = self(x, training=True)  # Forward pass
#             # Compute the loss value
#             # (the loss function is configured in `compile()`)
#             loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
#
#         print(loss)
#
#         # Compute gradients
#         trainable_vars = self.trainable_variables
#         print(trainable_vars)
#         gradients = tape.gradient(loss, trainable_vars)
#
#         print(gradients)
#
#         fisher = tf.matmul(gradients, tf.transpose(gradients))
#         fisher_inv = tf.matrix_inverse(fisher)
#         natural_grad = tf.matmul(fisher_inv, gradients)
#
#         # Update weights
#         self.optimizer.apply_gradients(zip(natural_grad, trainable_vars))
#         # Update metrics (includes the metric that tracks the loss)
#         self.compiled_metrics.update_state(y, y_pred)
#         # Return a dict mapping metric names to current value
#         return {m.name: m.result() for m in self.metrics}


# loss_tracker = keras.metrics.Mean(name="loss")
# mae_metric = keras.metrics.MeanAbsoluteError(name="mae")


class NaturalModel(keras.Model):
    def __init__(self, **kwargs):
        super(NaturalModel, self).__init__(**kwargs)
        self.layer_1 = Dense(101, activation='relu')
        self.layer_2 = Dense(50, activation='relu')
        self.layer_3 = Dense(50, activation='relu')
        self.layer_4 = Dense(2)

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        return x

    def log_normal(x, mu, var, eps=0.0, axis=-1):
        if eps > 0.0:
            var = tf.add(var, eps, name='clipped_var')
        return -0.5 * tf.reduce_sum(
            tf.math.log(2 * np.pi) + tf.math.log(var) + tf.square(x - mu) / var, axis)

    def get_loss(self, y, y_pred):
        print(y)
        print(y_pred)
        y_mu, y_sig = tf.split(y, 2, axis=1)
        y_pred_mu, y_pred_sig = tf.split(y, 2, axis=1)
        return tf.reduce_mean(self.log_normal(y_mu, y_sig) - self.log_normal(y_pred_mu, y_pred_sig))
        # return tf.reduce_mean(tfp.distributions.LogNormal(y_mu, y_sig) - tfp.distributions.LogNormal(y_pred_mu, y_pred_sig))

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        # print(x)
        # print(y)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

            y_mu, y_sig = tf.split(y, [1, 1], 1)
            y_pred_mu, y_pred_sig = tf.split(y_pred, [1, 1], 1)
            Y = tfp.distributions.Normal(loc=y_mu, scale=y_sig)
            Y_pred = tfp.distributions.Normal(loc=y_pred_mu, scale=y_pred_sig)
            kl_loss = tfp.distributions.kl_divergence(Y, Y_pred)
            print(kl_loss)
            # kl_loss = tf.keras.losses.kullback_leibler_divergence(Y, Y_pred)
            # kl_loss = tf.reduce_sum(Y * tf.math.log(Y/Y_pred))

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        print(gradients)

        def fisher(row):
            row = tf.reshape(row, [-1, 1])
            fisher = tf.matmul(row, tf.transpose(row))
            fisher_inv = tf.linalg.inv(fisher)
            natural_grad = tf.matmul(fisher_inv, row)
            return tf.reshape(natural_grad, [-1])

        natural_gradients = []
        bias = False
        for gradient in gradients:
            if not bias:
                # print(gradient)
                natural_gradients.append(tf.map_fn(fisher, gradient))
                bias = True
            else:
                natural_gradients.append(gradient)
                bias = False
        print(natural_gradients)

        # fisher = tf.matmul(gradients, tf.transpose(gradients)) # wrong dimensions
        # fisher_inv = tf.matrix_inverse(fisher)
        # natural_grad = tf.matmul(fisher_inv, gradients)

        # Update weights
        self.optimizer.apply_gradients(zip(natural_gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    seed(5)

    options = [(0, 0.2), (1, 0.6), (2, 1.0), (3, 1.4), (4, 1.8)]
    choice_list = np.random.randint(0, 5, 1030)

    dataset = []
    for i in range(1030):
        mu, sigma = options[choice_list[i]]
        sample = np.random.normal(mu, sigma, 400)
        hist = np.histogram(sample, 50)
        data = list(np.concatenate((hist[0], hist[1])))
        data.append(mu)
        data.append(sigma)
        dataset.append(data)

    df = pd.DataFrame(dataset)
    X = df.drop(columns=[101, 102], axis=1).values
    y = df[[101, 102]].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = NaturalModel()

    opt = RMSprop(learning_rate=0.0001)
    model.compile(optimizer=opt, loss='mse')

    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

    model.fit(x=X_train, y=y_train, epochs=600, validation_data=(X_test, y_test),
              callbacks=[early_stop])

    losses = pd.DataFrame(model.history.history)
    losses.plot()  # check for over fitting

    predictions = model.predict(X_test)
    pd.DataFrame(predictions)
    pred_df = pd.concat([pd.DataFrame(y_test), pd.DataFrame(predictions)], axis=1)
    print(pred_df)
