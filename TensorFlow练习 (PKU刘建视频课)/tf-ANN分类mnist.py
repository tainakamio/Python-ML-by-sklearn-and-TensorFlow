# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 16:47:05 2020

@author: Administrator
"""


#import numpy as np
import tensorflow as tf

n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32,shape=(None,n_inputs),name='X')
y = tf.placeholder(tf.int64,shape=(None),name='y')

with tf.name_scope('dnn'):
    l1 = tf.contrib.layers.fully_connected(X,n_hidden1,scope='hidden1')
    l2 = tf.contrib.layers.fully_connected(l1,n_hidden2,scope='hidden2')
    outputs = tf.contrib.layers.fully_connected(l2,n_outputs,scope='outputs')

with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y,logits=outputs)
    loss = tf.reduce_mean(xentropy,name='loss')
    tf.summary.scalar('loss',loss)

with tf.name_scope('train'):
    training_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    
with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(outputs,y,1)
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
    
init = tf.global_variables_initializer()
saver = tf.train.Saver()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/')

n_epochs = 3
batch_size = 50

from datetime import datetime
now = datetime.now().strftime('%Y%m%d%H%M%S')
root_logdir = 'tf_logs'
logdir = '{}/run-{}/'.format(root_logdir,now)

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for i in range(mnist.train.num_examples//batch_size):
            X_batch,y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op,feed_dict={X:X_batch,y:y_batch})
        acc_train = accuracy.eval(feed_dict={X:X_batch,y:y_batch})
        acc_test = accuracy.eval(feed_dict={X:mnist.test.images,
                                                y:mnist.test.labels})
        print(epoch,
              'Train accuracy:',acc_train,
              'Test accuracy:',acc_test)
#    saver.save(sess,'./my_model_final.ckpt')
    writer = tf.summary.FileWriter(logdir,tf.get_default_graph())
