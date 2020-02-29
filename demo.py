import argparse
import os, sys
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
from net.batch import *
from net.model import *


parser = argparse.ArgumentParser()
parser.add_argument("--test_dir", help = "Test directory Location")
parser.add_argument("--result_dir",default = os.path.join(".","FCN_Image") ,help = "Result directory Location")
parser.add_argument("--lr", nargs="?", default=1e-4, help = "Number of learning rate")
parser.add_argument("--keep_prob", nargs="?", default=0, help = "Number of keep prob")
parser.add_argument("--label", default = "..\Data\VOC2012\labels.txt", help = "location of labels")
parser.add_argument("--model", nargs="?", help ="Saved modle Location")

args = parser.parse_args()

test_dir = args.test_dir
name = os.listdir(test_dir)
img_list = [os.path.join(test_dir, i) for i in name]
re_dir = args.result_dir
keep_prob = args.keep_prob
label_path = args.label
labels, index = load_labels(label_path)
lr = args.lr
model = args.model


if not os.path.exists(re_dir):os.mkdir(re_dir)
if not os.path.exists(test_dir):raise TypeError("Please input right Train Data Path")

config = tf.ConfigProto()
config.gpu_options.allocator_type="BFC"
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

with tf.Session(config=config) as sess:

    fcn = FCN(sess, keep_prob, lr, len(labels)-1)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, model)

    imgs = image_load_resize(img_list)

    start = time.time()
    output = fcn.generate(imgs)
    for n,im in enumerate(output):
        plt.imsave(os.path.join(re_dir,name[n]), cmap=cm.Paired, arr= im)
    print("Done.. ", time.time()-start)
