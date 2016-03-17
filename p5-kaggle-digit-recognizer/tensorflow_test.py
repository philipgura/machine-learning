import tensorflow as tf

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from scipy.ndimage import convolve

import time

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def save_file(y_pred):
    y_pred_frame = pd.DataFrame(data=y_pred)
    y_pred_frame.index +=1
    y_pred_frame.columns = ['Label']

    y_pred_frame.to_csv(path_or_buf='data/test_labelsTensorFlow2.csv', sep=',', index=True, index_label='ImageId')
    print "Test data written!"

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

direction_vectors1 = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]],

        [[1, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 1],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 1]],

        [[0, 0, 0],
         [0, 0, 0],
         [1, 0, 0]],
        
        [[0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]]

def nudge_dataset(X, Y, direction_vectors):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 28x28 images in X around by 1px to left, right, down, up
    """
    start = time.time()
    shift = lambda x, w: convolve(x.reshape((28, 28)), mode='constant',
                                  weights=w).ravel()
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    
    Y = np.concatenate([Y for _ in range(13)], axis=0)
    end = time.time()

    print "Done!\nNudging time (secs): {:.3f}".format(end - start)
    
    return X, Y

train_data = tf.placeholder(tf.float32, shape=[None, 785])
train_data = pd.read_csv("data/train.csv", nrows=42000)
print "Train data loaded!"

#load and format final test data
test_data_final = tf.placeholder(tf.float32, shape=[None, 784])
test_data_final = pd.read_csv("data/test.csv")
print "Test data loaded!"

test_data_final = np.multiply(test_data_final, 1.0 / 255.0)
#####

feature_cols = list(train_data.columns[1:])
target_col = train_data.columns[0]

X_train_all = tf.placeholder(tf.float32, shape=[None, 784])
X_train = tf.placeholder(tf.float32, shape=[None, 784])
X_test = tf.placeholder(tf.float32, shape=[None, 784])
y_train_all = tf.placeholder(tf.int32, shape=[None, 1])

X_train_all = train_data[feature_cols]
y_train_all = train_data[target_col]

print "Feature Shape: "+str(X_train_all.shape)

y_train = tf.placeholder(tf.int32, shape=[None, 10])
y_test = tf.placeholder(tf.int32, shape=[None, 10])

y_train_hot = dense_to_one_hot(y_train_all, 10)
y_train_hot = y_train_hot.astype(np.uint8)

print "Label Shape: "+str(y_train_hot.shape)
print "Label Sample: "+str(y_train_hot[0])

#X_train_all, y_train_hot = nudge_dataset(X_train_all, y_train_hot, direction_vectors1)
#print "Feature Shape (nudge): "+str(X_train_all.shape)
#print "Label Shape (nudge): "+str(y_train_hot.shape)

#convert to [0:255] => [0.0:1.00]
X_train_all = np.multiply(X_train_all, 1.0 / 255.0) # (x - 128) / 128 try this

X_train, X_test, y_train, y_test = train_test_split(X_train_all, y_train_hot, test_size=2000, random_state=42)

index_in_epoch = 0
epochs_completed = 0
num_examples = X_train.shape[0]

def get_next_batch(i, batch_size):
    global X_train
    global y_train

    global index_in_epoch
    global epochs_completed
    global num_examples
    
    start = index_in_epoch
    index_in_epoch += batch_size
    
    # when all trainig data have been already used, it is reorder randomly    
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        X_train, y_train = shuffle(X_train, y_train)
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    
    return X_train[start:end], y_train[start:end]

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

batch_size = 50
num = X_train.shape[0]/batch_size
print "Number of iterations: "+str(num)

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

#for i in range(num):
#    batch_x, batch_y = get_next_batch(i, batch_size)
#    train_step.run(session=sess, feed_dict={x: batch_x, y_: batch_y})

#for i in range(1000):
#    batch = mnist.train.next_batch(50)
#    train_step.run(feed_dict={x: batch[0], y_: batch[1]})


#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#print("MNST Data Acurracy (test): "+str(accuracy.eval(session=sess, feed_dict={x: mnist.test.images, y_: mnist.test.labels})))
#print("My MNST Data Accuracy (test): "+str(accuracy.eval(session=sess, feed_dict={x: X_test, y_: y_test})))

#prediction = tf.argmax(y,1)

#y_pred = prediction.eval(session=sess, feed_dict={x: test_data_final})

#save_file(y_pred)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x, k):
  return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def norm(name, x, lsize=4):
  return tf.nn.lrn(x, lsize, bias=1.0, alpha=0.001/9.0, beta=0.75, name=name)


x_image = tf.reshape(x, [-1,28,28,1])
keep_prob = tf.placeholder(tf.float32)

# 1st Convolutional Layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool(h_conv1, k=2)

h_norm1 = norm('h_norm1', h_pool1, lsize=4)

# 2nd Convolutional Layer 1x1
W_conv2 = weight_variable([1, 1, 32, 32])
b_conv2 = bias_variable([32])

h_conv2 = tf.nn.relu(conv2d(h_norm1, W_conv2) + b_conv2)

# 3rd Convolutional Layer
W_conv3 = weight_variable([5, 5, 32, 64])
b_conv3 = bias_variable([64])

h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)
h_pool3 = max_pool(h_conv3, k=2)

h_norm2 = norm('h_norm2', h_pool3, lsize=4)

# 4th Convolutional Layer
#W_conv4 = weight_variable([5, 5, 64, 128])
#b_conv4 = bias_variable([128])

#h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
#h_pool4 = max_pool(h_conv4, k=2)

#h_norm3 = norm('h_norm3', h_pool4, lsize=4)

# 5th Convolutional Layer
#W_conv5 = weight_variable([3, 3, 128, 256])
#b_conv5 = bias_variable([256])

#h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)
#h_pool5 = max_pool(h_conv5, 2)

#h_norm4 = norm('h_norm3', h_pool5, lsize=4)

# 1st Densely Connnected Layer
fc1_nodes = 1024
W_fc1 = weight_variable([7 * 7 * 64, fc1_nodes])
b_fc1 = bias_variable([fc1_nodes])

h_pool3_flat = tf.reshape(h_norm2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

# Dropout
h_fc1 = tf.nn.dropout(h_fc1, keep_prob)

# 2nd Densely Connnected Layer
fc2_nodes = 1024
W_fc2 = weight_variable([fc1_nodes, fc2_nodes])
b_fc2 = bias_variable([fc2_nodes])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

# Dropout
h_fc2 = tf.nn.dropout(h_fc2, keep_prob)

# Readout Layer
fc3_nodes = 1024
W_fc3 = weight_variable([fc3_nodes, 10])
b_fc3 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)

# Train and Evaluate the Model
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

index_in_epoch = 0
epochs_completed = 0
num_examples = X_train.shape[0]

start = time.time()
for i in range(20001):
  batch_x, batch_y = get_next_batch(i, batch_size)
  if i%100 == 0:
    train_accuracy = accuracy.eval(session=sess, feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
    print("step %d, training accuracy %.5f"%(i, train_accuracy))
  train_step.run(session=sess, feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
end = time.time()
print "Done!\nTrain time (secs): {:.3f}".format(end - start)
print("test accuracy %.5f"%accuracy.eval(session=sess, feed_dict={x: X_test, y_: y_test, keep_prob: 1.0}))

prediction = tf.argmax(y_conv,1)
start = time.time()
y_pred = prediction.eval(session=sess, feed_dict={x: test_data_final, keep_prob: 1.0})
end = time.time()
print "Done!\nTest prediction time (secs): {:.3f}".format(end - start)

print y_pred[0]
print y_pred[1]
print y_pred[2]

save_file(y_pred)

sess.close()

