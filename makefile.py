import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import PIL
import numpy as np
import tempfile
from urllib.request import urlretrieve
import tarfile
import os
from PIL import Image
import json
import matplotlib.pyplot as plt
import matplotlib.image as mp
import glob

tf.logging.set_verbosity(tf.logging.ERROR)
sess = tf.InteractiveSession()
image = tf.Variable(tf.zeros((224, 224, 3)))


def inception(image, reuse):
    preprocessed = tf.multiply(tf.subtract(tf.expand_dims(image, 0), 0.5), 2.0)
    arg_scope = nets.inception.inception_v1_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
        logits, _ = nets.inception.inception_v1(
            preprocessed, 1001, is_training=False, reuse=reuse)
        logits = logits[:, 1:]  # ignore background class
        probs = tf.nn.softmax(logits)  # probabilities
    return logits, probs


logits, probs = inception(image, reuse=False)

data_dir = tempfile.mkdtemp()
inception_tarball, _ = urlretrieve(
    'http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz')
tarfile.open(inception_tarball, 'r:gz').extractall(data_dir)
restore_vars = [
    var for var in tf.global_variables()
    if var.name.startswith('InceptionV1/')
]
saver = tf.train.Saver(restore_vars)
saver.restore(sess, os.path.join(data_dir, 'inception_v1.ckpt'))

imagenet_json, _ = urlretrieve(
    'http://www.anishathalye.com/media/2017/07/25/imagenet.json')
with open(imagenet_json) as f:
    imagenet_labels = json.load(f)


def classify(img, correct_class=None, target_class=None, label='o'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    fig.sca(ax1)
    p = sess.run(probs, feed_dict={image: img})[0]
    ax1.imshow(img)
    fig.sca(ax1)
    topk = list(p.argsort()[-10:][::-1])
    topprobs = p[topk]
    print(topprobs)
    barlist = ax2.bar(range(10), topprobs)
    for t in topk:
        print(topk.index(t))
        barlist[topk.index(t)].set_color('r')
    for i in topk:
        print(topk.index(i))
        barlist[topk.index(i)].set_color('g')
    plt.sca(ax2)
    plt.ylim([0, 1.1])
    plt.xticks(range(10),
               [imagenet_labels[i][:15] for i in topk],
               rotation='vertical')
    fig.subplots_adjust(bottom=0.2)
    plt.show()

def slove(img_path,demo_steps):
    img = PIL.Image.open(img_path)
    wide = img.width > img.height
    new_w = 224 if not wide else int(img.width * 224 / img.height)
    new_h = 224 if wide else int(img.height * 224 / img.width)
    img = img.resize((new_w, new_h)).crop((0, 0, 224, 224))
    img = (np.asarray(img) / 255.0).astype(np.float32)
    x = tf.placeholder(tf.float32, (224, 224, 3))
    x_hat = image  # our trainable adversarial input
    assign_op = tf.assign(x_hat, x)
    learning_rate = tf.placeholder(tf.float32, ())
    y_hat = tf.placeholder(tf.int32, ())
    labels = tf.one_hot(y_hat, 1000)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=[labels])
    optim_step = tf.train.GradientDescentOptimizer(
        learning_rate).minimize(loss, var_list=[x_hat])
    epsilon = tf.placeholder(tf.float32, ())
    below = x - epsilon
    above = x + epsilon
    projected = tf.clip_by_value(tf.clip_by_value(x_hat, below, above), 0, 1)
    with tf.control_dependencies([projected]):
        project_step = tf.assign(x_hat, projected)
    demo_epsilon = 2.0 / 255.0  # a really small perturbation
    demo_lr = 1e-2
    demo_target = 58  # "水蛇"
    # 初始化
    sess.run(assign_op, feed_dict={x: img})
    for i in range(demo_steps):
        # 梯度下降
        _, loss_value = sess.run(
            [optim_step, loss],
            feed_dict={learning_rate: demo_lr, y_hat: demo_target})
        # project step
        sess.run(project_step, feed_dict={x: img, epsilon: demo_epsilon})
        if (i + 1) % 5 == 0:
            print('step %d, loss=%g' % (i + 1, loss_value))
    adv = x_hat.eval()  # retrieve the adversarial example
    mp.imsave('mnt//data//aigroup-data//yanh-data//pic//40//n01484850//' + img_path[51:], adv)

paths = glob.glob('mnt//data//aigroup-data//yanh-data//val//n01484850//*')
for path in paths:
    slove(path,40)
