import tensorflow as tf
import os,sys
import numpy as np
import time
from net.vgg16 import *


class FCN():
    
    def __init__(self, sess, keep=0.5, learning_rate=1e3, n_class=21, is_train=True):
        
        self.sess = sess
        self.keep_prob = keep
        self.lr = learning_rate
        self.n_class = n_class
        self.is_train = is_train
        self.vgg = Vgg16()
        self.__build_net__()
        
    def __build_net__(self):

        print("Build Strat..")
        start_time = time.time()

        self.x = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.y = tf.placeholder(tf.int64, [None, 224, 224])
        self.is_train = tf.placeholder(tf.bool)

        with tf.name_scope("VGG"):
            self.vgg.build(self.x)

        x = self.vgg.pool5
        x_32 = self.vgg.pool4
        x_16 = self.vgg.pool3

        h = self.dropout(self.relu(self.conv2d(x, 4096, 1, 1, "SAME", "Convolutional1")), self.keep_prob, self.is_train)
        h = self.dropout(self.relu(self.conv2d(h, 4096, 1, 1, "SAME", "Convolutional2")), self.keep_prob, self.is_train)
        h = self.conv2d(h, self.n_class, 1, 1, "SAME", "Convolutional")

        h = self.upconv2d(h, self.n_class, 4, 2, "upconv1")
        x_32 = self.relu(self.conv2d(x_32,self.n_class,1,1,"SAME","AddConv1"))
        h = tf.add(h, x_32)

        h = self.upconv2d(h, self.n_class, 4, 2, "upconv2")
        x_16 = self.relu(self.conv2d(x_16,self.n_class,1,1,"SAME","AddConv2"))
        h = tf.add(h, x_16)

        h = self.upconv2d(h, self.n_class, 16, 8, "last_out")

        self.output = tf.argmax(h,3)

        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.output, self.y), tf.float32))
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.y, logits=h))
        self.train_op = tf.train.AdamOptimizer(learning_rate = self.lr).minimize(self.loss)

        print("Build Done..")
        print("Build Time : ", time.time() - start_time)
    

    def conv2d(self, x, out, kernel, stride, padding="SAME", name=None):
        with tf.variable_scope(name):
            return tf.layers.conv2d(x, out, kernel, stride, padding=padding)

    def upconv2d(self, x, out, kernel, stride, name=None):
        with tf.variable_scope(name):
            return tf.layers.conv2d_transpose(x, out, kernel, stride, padding="SAME")

    def relu(self, x):
        return tf.nn.relu(x)

    def training(self, image, gt, is_train = True):
        return self.sess.run([self.loss, self.acc, self.train_op], feed_dict = {self.x:image,
                                                                                self.y:gt,
                                                                                self.is_train:is_train})

    def testing(self, image, gt, is_train = False):
        return self.sess.run([self.loss, self.acc], feed_dict = {self.x:image,
                                                                 self.y:gt,
                                                                 self.is_train:is_train})

    def generate(self, image, is_train = False):
        return self.sess.run(self.output, feed_dict = {self.x:image,
                                                       self.is_train:is_train})


    def dropout(self, x, keep_prob=0.5, is_train=True):
        return tf.layers.dropout(x, keep_prob, training=is_train)
    
    
