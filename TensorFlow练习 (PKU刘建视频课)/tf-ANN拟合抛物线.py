# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 20:17:07 2020

@author: Administrator
"""


import numpy as np
import  tensorflow as tf
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

x_data = np.linspace(-1,1,300)[:,np.newaxis]
#后面那个东西把原本(300,)的1维数组变成(300,1)的2维数组
noise = np.random.normal(0,0.01,x_data.shape)
y_data = np.square(x_data)+noise

xs = tf.placeholder(tf.float32,x_data.shape)
ys = tf.placeholder(tf.float32,y_data.shape)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x_data,y_data,s=3)
plt.ylim(-0.1,1.1)
plt.ion()

def add_layer(inputs,in_size,out_size,n_layer,act_fn=None):
    layer_name = 'layer%s'%n_layer
    with tf.name_scope('layer_name'):
        with tf.name_scope('weights'):            
            Weights = tf.Variable(tf.random_normal([in_size,out_size]))
            tf.summary.histogram(layer_name+'/weights',Weights)
        with tf.name_scope('biases'):
            bias = tf.Variable(tf.zeros([1,out_size]))
        with tf.name_scope('wx_plus_b'):
            WX_plus_b = tf.matmul(inputs,Weights)+bias
            tf.summary.histogram(layer_name+'biases',bias)
        if act_fn == None:
            outputs = WX_plus_b
        else:
            outputs = act_fn(WX_plus_b)
        tf.summary.histogram(layer_name+'wx_plus_b',WX_plus_b)
        return outputs

l1 = add_layer(xs,1,100,n_layer=1,act_fn=tf.nn.relu)
pred = add_layer(l1,100,1,n_layer=2,act_fn = None)

loss = tf.reduce_mean(tf.square(pred-ys),reduction_indices=0)
#不写reduction_indices=0,也可。都能压缩成一个标量。
#但是不能写reduction_indices=1.
train = tf.train.MomentumOptimizer(0.01,0.5).minimize(loss)

from datetime import datetime
now = datetime.now().strftime('%Y%m%d%H%M%S')
root_logdir = 'tf_logs'
logdir = '{}/run-{}/'.format(root_logdir,now)


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(logdir,tf.get_default_graph())
    for i in range(1001):
        sess.run(train,feed_dict={xs:x_data,ys:y_data})
        if i%10==0:
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction = sess.run(pred,feed_dict={xs:x_data})
#prediction只run第二层，不涉及ys，只要给xs喂值即可。
            lines = ax.plot(x_data,prediction,'r-',lw=1)
            plt.pause(0.1)
        # if i%50==0:
        #     result = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        #     writer.add_summary(result,i)
            
