# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 15:39:07 2020

@author: Administrator
"""


# import sklearn as skl
# import pandas as pd
# from pandas import  DataFrame
from sklearn import datasets
import numpy as np
import tensorflow as tf

x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

# x_data = DataFrame(x_data,columns=['calyx_l','calyx_w','ptl_l','ptl_w'])
# x_data['catagory'] = y_data

np.random.seed(1)
np.random.shuffle(x_data)
np.random.shuffle(y_data)

x_train = x_data[:-30]
y_train = y_data[:-30]
x_tst = x_data[-30:]
y_tst = y_data[-30:]

train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(32)
tst_db = tf.data.Dataset.from_tensor_slices((x_tst,y_tst)).batch(32)

w1 = tf.Variable(tf.random.truncated_normal([4,3],stddev=0.1,seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3],stddev=0.1,seed=1))

w1 = tf.cast(w1,dtype=tf.float32)

lr = 0.1
train_loss_results = []
test_acc = []
epoch = 500
loss_all = 0

x_train = tf.cast(x_train,dtype=tf.float32)
x_tst = tf.cast(x_tst,dtype=tf.float32)


for epoch in range(epoch):
    for step,(x_train,y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train,w1)+b1
            y = tf.nn.softmax(y)
            y_ = tf.one_hot(y_train,depth=3)
            loss = tf.reduce_mean(tf.square(y_-y))
            loss_all += loss.numpy()
        grads = tape.gradient(loss,[w1,b1])
        w1.assign_sub(lr*grads[0])
        b1.assign_sub(lr*grads[1])
    print('Epoch{},loss{}'.format(epoch,loss_all/4))
 
total_correct,total_number = 0,0
for x_tst,y_tst in tst_db:
    y = tf.matmul(x_tst,w1)+b1
    y = tf.nn.softmax(y)
    pred = tf.argmax(y,axis=1)
    pred = tf.cast(pred,dtype=y_tst.dtype)
    correct = tf.cast((tf.equal(pred,y_tst)),dtype=tf.int32)
    correct = tf.reduce_sum(correct)
    total_correct += int(correct)
    total_number += x_tst.shape[0]
acc = total_correct/total_number
print(acc)

