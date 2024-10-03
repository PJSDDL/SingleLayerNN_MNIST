import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

#载入数据集
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
#设置批次的大小
batch_size=100
#计算一共有多少个批次
n_batch=mnist.train.num_examples//batch_size

#定义初始化权值函数
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

#定义初始化偏置函数
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#图像resize
def batch_resize(input_img):
    out_img = np.zeros((batch_size, 256))
    for index in range(batch_size):
        reshape_img = np.reshape(input_img[index], (28, 28))
        resize_img = cv.resize(reshape_img, (16, 16))
        out_img[index, :] = np.reshape(resize_img, 256)

    return out_img

#输入层
#定义两个placeholder
x=tf.placeholder(tf.float32,[None,16*16]) #16*16
y=tf.placeholder(tf.float32,[None,10])

#全连接
W = weight_variable([16*16,10])
b = bias_variable([10])

#输出层
#计算输出
prediction=tf.nn.softmax(tf.matmul(x,W)+b)

#交叉熵代价函数
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

#使用AdamOptimizer进行优化
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#结果存放在一个布尔列表中(argmax函数返回一维张量中最大的值所在的位置)
correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))

#求准确率(tf.cast将布尔值转换为float型)
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#创建会话
with tf.Session() as sess:
    start_time=time.clock()
    sess.run(tf.global_variables_initializer()) #初始化变量
    for epoch in range(12): #迭代12次（训练12次）
        for batch in range(n_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            batch_xs_re = batch_resize(batch_xs)
            sess.run(train_step,feed_dict={x:batch_xs_re, y:batch_ys}) #进行迭代训练
        #计算准确率
        acc=sess.run(accuracy,feed_dict={x:batch_xs_re, y:batch_ys})
        print('Iter'+str(epoch)+',Testing Accuracy='+str(acc))
    end_time=time.clock()
    print('Running time:%s Second'%(end_time-start_time)) #输出运行时间

    np.savetxt('NUM1.txt', batch_xs_re[0], delimiter = ', ', newline = ',\n', header = "NUM1 = {", footer = "};")
    np.savetxt('NUM2.txt', batch_xs_re[1], delimiter = ', ', newline = ',\n', header = "NUM2 = {", footer = "};")
    np.savetxt('NUM3.txt', batch_xs_re[2], delimiter = ', ', newline = ',\n', header = "NUM3 = {", footer = "};")
    np.savetxt('NUM4.txt', batch_xs_re[3], delimiter = ', ', newline = ',\n', header = "NUM4 = {", footer = "};")
    np.savetxt('NUM5.txt', batch_xs_re[4], delimiter = ', ', newline = ',\n', header = "NUM5 = {", footer = "};")
    np.savetxt('NUM6.txt', batch_xs_re[5], delimiter = ', ', newline = ',\n', header = "NUM6 = {", footer = "};")

    W = sess.run(W,feed_dict={x:batch_xs_re,y:batch_ys})
    np.savetxt('W.txt', W, delimiter = ', ', newline = ',\n', header = "W = {", footer = "};")
    b = sess.run(b,feed_dict={x:batch_xs_re,y:batch_ys})
    np.savetxt('b.txt', b, delimiter = ', ', newline = ',\n', header = "b = {", footer = "};")
