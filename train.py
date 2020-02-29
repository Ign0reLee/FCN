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


parser = argparse.ArgumentParser(
    usage="""train.py --train_dir TrainDirectory --train_gt_dir TrainGroundTruthDirectory --test.py TestDirectory --test_gt_dir TestGroundTruthDirectory [Option]""")
parser.add_argument("--train_dir", help = "Train directory Location")
parser.add_argument("--train_gt_dir", help = "Train ground truth Location")
parser.add_argument("--test_dir", help = "Test directory Location")
parser.add_argument("--test_gt_dir", help = "Test ground truth Location")
parser.add_argument("--batch_size", nargs="?", default = 8, help = "Number of batch_size")
parser.add_argument("--label", default = "..\Data\VOC2012\labels.txt", help = "location of labels")
parser.add_argument("--epoch", nargs="?", default=20, help = "Number of Training epochs")
parser.add_argument("--lr", nargs="?", default=1e-4, help = "Number of learning rate")
parser.add_argument("--keep_prob", nargs="?", default=0.85, help = "Number of keep prob")
parser.add_argument("--save_dir", nargs="?", default="./model/", help="Location of saved model directory")


args = parser.parse_args()


train_path = args.train_dir
train_gt_path = args.train_gt_dir
test_path = args.test_dir
test_gt_path = args.test_gt_dir
save_path = args.save_dir
label_path = args.label
keep_prob = float(args.keep_prob)
lr = float(args.lr)
epochs = int(args.epoch)
batch_size = int(args.batch_size)

if not os.path.exists(save_path):os.mkdir(save_path)
if not os.path.exists(os.path.join(".","result")):os.mkdir(os.path.join(".","result"))
if not os.path.exists(os.path.join(".","result_img")):os.mkdir(os.path.join(".","result_img"))
if not os.path.exists(os.path.join(".","result_img","val")):os.mkdir(os.path.join(".","result_img","val"))
if not os.path.exists(train_path):raise TypeError("Please input right Train Data Path")
if not os.path.exists(train_gt_path):raise TypeError("Please input right Train Ground Truth Data Path")
if not os.path.exists(test_path):raise TypeError("Please input right Test Data Path")
if not os.path.exists(test_gt_path):raise TypeError("Please input right Test Ground Truth Data Path")


trlist, t_size, val_size = load_path_list(train_path, train_gt_path, batch_size)
valist = trlist[t_size: t_size+val_size]
trlist = trlist[:t_size]
telist, _, _ = load_path_list(test_path, test_gt_path, batch_size, False)
labels, index = load_labels(label_path)

train_loss = []
train_acc = []
train_steps = []
val_loss =[]
val_acc = []
val_steps =[]

config = tf.ConfigProto()
config.gpu_options.allocator_type="BFC"
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

with tf.Session(config=config) as sess:

    fcn = FCN(sess, keep_prob, lr, len(labels)-1)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    saver = tf.train.Saver()

    for epoch in range(epochs):
        
        print("Data Shuffle..")
        rd.shuffle(trlist)
        print("Shuffle Done..")

        batches = batch(train_path, train_gt_path, trlist, batch_size, t_size, index)

        for i, now in enumerate(batches):

            img, gt = now
            start = time.time()
            loss, acc, _ = fcn.training(image = img, gt = gt)

            train_loss.append(loss)
            train_acc.append(acc)
            train_steps.append(str(epoch)+"_"+str(i))
            
            print("Epoch : ", epoch, " Step : ", i, "Loss : ",loss, "ACC : ", acc*100, "% time : ", time.time()-start)

            if i % 101 == 0:

                print("Validation Testing..")
                test = batch(train_path, train_gt_path, valist, len(valist), val_size, index)
                test_img, test_gt = next(test)
                start = time.time()
                loss, acc = fcn.testing(test_img, test_gt)
                print("Validation Result.. ValLoss : ",loss, "ValACC : ", acc*100, "% time : ", time.time()-start)

                print("Validation Image result making..")
                start = time.time()
                output = fcn.generate(test_img)
                for n,im in enumerate(output):
                    plt.imsave(os.path.join(".","result_img","val",str(epoch)+"_"+str(i)+"_"+str(n)+".jpg"), cmap=cm.Paired, arr= im)
                print("Done.. ", time.time()-start)
                val_loss.append(loss)
                val_acc.append(acc)
                val_steps.append(str(epoch)+"_"+str(i))
                saver.save(sess, os.path.join(save_path, str(epoch)+"Epoch_"+str(i)+"Step.ckpt"))
                

        print("Validation Testing..")
        test = batch(train_path, train_gt_path, valist, len(valist), val_size, index)
        test_img, test_gt = next(test)
        start = time.time()
        loss, acc = fcn.testing(test_img, test_gt)
        print("Validation Result.. ValLoss : ",loss, "ValACC : ", acc*100, "% time : ", time.time()-start)

        print("Validation Image result making..")
        start = time.time()
        output = fcn.generate(test_img)
        for n,im in enumerate(output):
            plt.imsave(os.path.join(".","result_img","val",str(epoch)+"_"+str(i)+"_"+str(n)+".jpg"), cmap=cm.Paired, arr= im)
        print("Done.. ", time.time()-start)
        val_loss.append(loss)
        val_acc.append(acc)
        val_steps.append(str(epoch)+"_"+str(i))
        saver.save(sess, os.path.join(save_path, str(epoch)+"Epoch_"+str(i)+"Step.ckpt"))

fig1, ax1 = plt.subplots()
lines, = ax1.plot(train_steps,train_loss)
fig1.savefig(os.path.join(".","result","Train_Loss_Grpah"), dpi=300)
plt.close(fig1)

fig2, ax2 =  plt.subplots()
lines, = ax2.plot(train_steps,train_acc)
fig2.savefig(os.path.join(".","result","Train_Acc_Grpah"), dpi=300)
plt.close(fig2)

fig3, ax3 = plt.subplots()
lines, = ax3.plot(val_steps,val_loss)
fig3.savefig(os.path.join(".","result","Validation_Loss_Grpah"), dpi=300)
plt.close(fig3)

fig4, ax4 = plt.subplots()
lines, = ax4.plot(val_steps,val_acc)
fig4.savefig(os.path.join(".","result","Validation_Acc_Grpah"), dpi=300)
plt.close(fig4)
