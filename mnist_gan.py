from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

num_steps = 10000
batch_size = 128
learning_rate = 0.0002
display_steps = 1000

image_dim = 784
gen_hidden_dim = 256
disc_hidden_dim = 256
noise_dim = 100
model_path = "tmp/model.ckpt"

def glorot_init(shape):
    return tf.random_normal(shape, stddev=1./tf.sqrt(shape[0] / 2.))

weights = {
    "gen_hidden": tf.Variable(glorot_init([noise_dim, gen_hidden_dim])),
    "gen_out": tf.Variable(glorot_init([gen_hidden_dim, image_dim])),
    "disc_hidden": tf.Variable(glorot_init([image_dim, disc_hidden_dim])),
    "disc_out": tf.Variable(glorot_init([disc_hidden_dim, 1]))
}

biases = {
    "gen_hidden": tf.Variable(tf.zeros([gen_hidden_dim])),
    "gen_out": tf.Variable(tf.zeros([image_dim])),
    "disc_hidden": tf.Variable(tf.zeros([disc_hidden_dim])),
    "disc_out": tf.Variable(tf.zeros([1]))
}

def generator(x):
    layer1 = tf.nn.relu(tf.add(tf.matmul(x, weights['gen_hidden']), biases['gen_hidden']))
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['gen_out']), biases['gen_out']))
    return layer2

def discriminator(x):
    layer1 = tf.nn.relu(tf.add(tf.matmul(x, weights['disc_hidden']), biases['disc_hidden']))
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['disc_out']), biases['disc_out']))
    return layer2

disc_input = tf.placeholder(tf.float32, [None, image_dim])
gen_input = tf.placeholder(tf.float32, [None, noise_dim])

gen_sample = generator(gen_input)

disc_real = discriminator(disc_input)
disc_fake = discriminator(gen_sample)

gen_loss = - tf.reduce_mean(tf.log(disc_fake))
disc_loss = - tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))

gen_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
disc_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

gen_vars = [weights['gen_hidden'], weights['gen_out'], biases['gen_hidden'], biases['gen_out']]
disc_vars = [weights['disc_hidden'], weights['disc_out'], biases['disc_hidden'], biases['disc_out']]

gen_train_op = gen_optimizer.minimize(gen_loss, var_list=gen_vars)
disc_train_op = disc_optimizer.minimize(disc_loss, var_list=disc_vars)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    saver.restore(sess, model_path)
    print("Model restored from ", model_path)

    for i in range(num_steps + 1):
        batch_x, _ = mnist.train.next_batch(batch_size)
        noise_ = np.random.uniform(-1., 1., [batch_size, noise_dim])
        _, _, gl, dl = sess.run([gen_train_op, disc_train_op, gen_loss, disc_loss],
                                               feed_dict={disc_input: batch_x, gen_input: noise_})
        if i % display_steps == 0:
            sess.as_default()
            print("Step: " + str(i) + " Generator loss: " + str(gl) + " Discriminator loss: " + str(dl))

    save_path = saver.save(sess, model_path)
    print("Model saved in file")
    f, a = plt.subplots(4, 10, figsize=(10, 4))
    for i in range(10):
        # Noise input.
        z = np.random.uniform(-1., 1., size=[4, noise_dim])
        g = sess.run([gen_sample], feed_dict={gen_input: z})
        g = np.reshape(g, newshape=(4, 28, 28, 1))
        # Reverse colours for better display
        g = -1 * (g - 1)
        for j in range(4):
            # Generate image from noise. Extend to 3 channels for matplot figure.
            img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2),
                             newshape=(28, 28, 3))
            a[j][i].imshow(img)

    f.show()
    plt.draw()
    plt.waitforbuttonpress()




