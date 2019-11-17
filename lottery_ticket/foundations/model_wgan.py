import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from lottery_ticket.foundations import model_base

class ModelWgan(model_base.ModelBase):
    
    def __init__(self,
                 hyperparameters,
                 input_placeholder,
                 label_placeholder,
                 presets=None,
                 masks=None):

        super(ModelWgan, self).__init__(presents=presets, masks=masks)

        def xavier_init(size):
            in_dim = size[0]
            xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
            return tf.random_normal(shape=size, stddev=xavier_stddev)
       
        # dimensions
        mb_size = 32
        X_dim = 784
        z_dim = 10
        h_dim = 128
        lam = 10
        n_disc = 5
        self.lr = 1e-4
        
        # variable creation
        X = input_placeholder[0]
        z = input_placeholder[1]

        # create generator
        G_W1 = self.dense_layer(
                'g1',
                z,
                h_dim,
                activation=tf.nn.relu,
                kernel_initializer=xavier_init
        )

        G_sample = self.dense_layer(
                'g2',
                G_W1,
                X_dim,
                activation=tf.nn.sigmoid,
                kernel_initializer=xavier_init
        )

        self.theta_G = ['g1_w', 'g1_b', 'g2_w', 'g2_b']

        D_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
        D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

        D_W2 = tf.Variable(xavier_init([h_dim, 1]))
        D_b2 = tf.Variable(tf.zeros(shape=[1]))

        self.theta_D = [D_W1, D_b1, D_W2, D_b2]

        # create discriminator
        def D(X):
            D_h1 = tf.nn.relu(tf.matmul(X, D_W1) + D_b1)
            out = tf.matmul(D_h1, D_W2) + D_b2
            return out

        D_real = D(X)
        D_fake = D(G_sample)
        
        eps = tf.random_uniform([mb_size, 1], minval=0., maxval=1.)
        X_inter = eps*X + (1. - eps)*G_sample
        grad = tf.gradients(D(X_inter), [X_inter])[0]
        grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
        grad_pen = lam * tf.reduce_mean((grad_norm - 1)**2)

        self.D_loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real) + grad_pen
        self.G_loss = -tf.reduce_mean(D_fake)

    def sample_z(m, n):
        return np.random.uniform(-1., 1., size=[m, n])








mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig





X = tf.placeholder(tf.float32, shape=[None, X_dim])


z = tf.placeholder(tf.float32, shape=[None, z_dim])

G_sample = G(z)
D_real = D(X)
D_fake = D(G_sample)

eps = tf.random_uniform([mb_size, 1], minval=0., maxval=1.)
X_inter = eps*X + (1. - eps)*G_sample
grad = tf.gradients(D(X_inter), [X_inter])[0]
grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
grad_pen = lam * tf.reduce_mean((grad_norm - 1)**2)

D_loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real) + grad_pen
G_loss = -tf.reduce_mean(D_fake)

D_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
            .minimize(D_loss, var_list=theta_D))
G_solver = (tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
            .minimize(G_loss, var_list=theta_G))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

for it in range(1000000):
    for _ in range(n_disc):
        X_mb, _ = mnist.train.next_batch(mb_size)

        _, D_loss_curr = sess.run(
            [D_solver, D_loss],
            feed_dict={X: X_mb, z: sample_z(mb_size, z_dim)}
        )

    _, G_loss_curr = sess.run(
        [G_solver, G_loss],
        feed_dict={z: sample_z(mb_size, z_dim)}
    )

    if it % 1000 == 0:
        print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'
              .format(it, D_loss_curr, G_loss_curr))

        if it % 1000 == 0:
            samples = sess.run(G_sample, feed_dict={z: sample_z(16, z_dim)})

            fig = plot(samples)
            plt.savefig('out/{}.png'
                        .format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)
