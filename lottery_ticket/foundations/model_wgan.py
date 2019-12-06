# Copyright (C) 2018 Google Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A GAN model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lottery_ticket.foundations import model_base
import tensorflow as tf
import inspect


class ModelWgan(model_base.ModelBase):
  """A GAN with user-specifiable hyperparameters."""

  def __init__(self,
               hyperparameters,
               input_placeholder,
               label_placeholder,
               presets=None,
               masks=None):
    """Creates a GAN.

    Args:
      hyperparameters: A dictionary of hyperparameters for the network.
        For this class, a single hyperparameter is available: 'layers'. This
        key's value is a list of (# of units, activation function) tuples
        for each layer in order from input to output. If the activation
        function is None, then no activation will be used.
      input_placeholder: A placeholder for the network's input.
      label_placeholder: A placeholder for the network's expected output.
      presets: Preset initializations for the network as in model_base.py
      masks: Masks to prune the network as in model_base.py.
    """
    # Call parent constructor
    super(ModelWgan, self).__init__(presets=presets, masks=masks)

    # Define dimensions
    mb_size = 32
    X_dim, z_dim, h_dim = 784, 10, 128
    lam = 10
    n_disc = 5
    self.lr = 1e-4

    # Variable creation
    X = input_placeholder
    z = tf.placeholder(tf.float32, shape=[None, z_dim])

    def xavier_init(size):
      in_dim = size[0]
      xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
      return tf.random_normal(shape=size, stddev=xavier_stddev)

    # Create generator
    G_W1 = self.dense_layer('g1', z, h_dim, activation=tf.nn.relu, 
                            # kernel_initializer=xavier_init)
                            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    G_sample = self.dense_layer('g2', G_W1, X_dim, activation=tf.nn.sigmoid,
                                # kernel_initializer=xavier_init)
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    self.theta_G = ['g1_w', 'g1_b', 'g2_w', 'g2_b']

    # Create discriminator
    # D_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
    # D_W1 = tf.Variable(tf.contrib.layers.xavier_initializer([X_dim, h_dim]))
    D_W1 = tf.get_variable('D_W1', [X_dim, h_dim], tf.contrib.layers.xavier_initializer(uniform=False))
    D_b1 = tf.Variable(tf.zeros(shape=[hdim]))

    # D_W2 = tf.Variable(xavier_init([h_dim, 1]))
    D_W2 = tf.Variable(tf.contrib.layers.xavier_initializer([h_dim, 1]))
    D_b2 = tf.Variable(tf.zeros(shape=[1]))

    self.theta_D = [D_W1, D_b1, D_W2, D_b2]

    def D(X):
      D_h1 = tf.nn.relu(tf.matmul(X, D_W1) + D_b1)
      out = tf.matmul(D_h1, D_W2) + D_b2
      return out 

    D_real = D(X)
    D_fake = D(G_sample)
    eps = tf.random_uniform([mb_size, 1], minval=0., maxval=1.)
    X_inter = eps * X + (1. - eps) * G_sample
    grad = tf.gradients(D(X_inter), [X_inter])[0]
    grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
    grad_pen = lam * tf.reduce_mean((grad_norm - 1)**2)

    self.D_loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real) + grad_pen
    self.G_loss = -tf.reduce_mean(D_fake)

    # TODO create generator loss

  def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])






