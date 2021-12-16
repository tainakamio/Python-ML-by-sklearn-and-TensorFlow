# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 21:07:32 2020

@author: Administrator
"""
#随机生成100个服从U[0,1)的样本值，用tf作GD线性回归

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

X_data = np.random.rand(100).astype(np.float32)
noise = np.random.normal(0,0.25,X_data.shape)
y_data = 4.0*X_data+3.0+noise

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X_data,y_data)
plt.ion()
#plt.show()

weights = tf.Variable(tf.random.uniform([1],-1.0,1.0))
#bias = tf.Variable(np.zeros([1]).astype(np.float32))
#不把bias设为float32，则下面的wx+b会报错，说bias是64位，而wx是32位，无法相加
bias = tf.Variable(tf.zeros([1]))
#这样写的话天然就是float32的，不必转换

y = weights*X_data+bias
loss = tf.reduce_mean(tf.square(y-y_data))
train = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    
    for step in range(1001):
        sess.run(train)
        if step%20==0:
#            print(step,sess.run(weights),sess.run(bias))
             try:
                 ax.lines.remove(lines[0])
             except Exception:
                 pass
             prediction = sess.run(y)
             lines = ax.plot(X_data,prediction,'r-',lw=1)
             plt.pause(0.5)
          
#步数在range（100）时并不会打出第100次的结果；
#要显示第100次的结果，步数起码要101
            
            