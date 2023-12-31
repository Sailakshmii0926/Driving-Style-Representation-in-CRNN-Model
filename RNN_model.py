"""
This is an implementation of RNN architecture which is presented in "characterizing driving styles with deep learning", Dong et al. (2016). 
Author: Sobhan Moosavi
"""

from __future__ import division
from __future__ import print_function

import argparse
import functools
import os
import random
import shutil
import time

import cPickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from tensorflow.contrib import rnn

# some parameters as input
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=125)
parser.add_argument('--neurons', type=int, default=100)
args = parser.parse_args()
epochs    = args.epochs


# helper class
def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class RNN_MODEL:

    def __init__(self, data, target, dropout, num_layers, timesteps=128):
        self.data = data
        self.target = target
        self.dropout = dropout
        self.num_layers = num_layers
        self._num_hidden = args.neurons
        self._timesteps = timesteps
        self.prediction
        self.error
        self.optimize
        self.accuracy
        self.predProbs

    @lazy_property
    def prediction(self):
        # Recurrent network.
        stacked_rnn = []
        for i in range(self.num_layers):
            cell = rnn.BasicLSTMCell(num_units=self._num_hidden, state_is_tuple=True, forget_bias=1.0)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0-self.dropout[i])
            stacked_rnn.append(cell)            
        network = tf.contrib.rnn.MultiRNNCell(cells=stacked_rnn, state_is_tuple=True)
        
        x = tf.unstack(self.data, self._timesteps, 1)
        output, _ = rnn.static_rnn(network, x, dtype=tf.float32)
        
        # Softmax layer parameters
        weight, bias = self._weight_and_bias(self._num_hidden, int(self.target.get_shape()[1]))
        
        #Embedding
        embedding = tf.matmul(output[-1], tf.Variable(np.identity(self._num_hidden, dtype="float32")))
        
        # Linear activation, using rnn inner loop last output    
        logits = tf.matmul(output[-1], weight) + bias
        soft_reg = tf.nn.softmax(logits)
        return soft_reg  
    
    @lazy_property
    def cost(self):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.target * tf.log(self.prediction), reduction_indices=[1]))  
        return cross_entropy

    @lazy_property
    def optimize(self):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.00005, momentum=0.9, epsilon=1e-6)        
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))
    
    @lazy_property
    def accuracy(self):
        correct_pred = tf.equal(tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)
        
    @lazy_property
    def predProbs(self):
        softmax_prob = self.prediction
        return softmax_prob


# to load pre-constructed feature matrices
def load_data(file):
    trip_segments = np.load(file)#/40.0
    print("Number of samples: {}".format(trip_segments.shape[0]))
    return trip_segments
        
        
# to split data to train, dev, and test; default: 75% train, 10% dev, and 15% test
def returnTrainDevTestData():

    matrices = load_data('data/RandomSample_5_10.npy')
    keys = cPickle.load(open('data/RandomSample_5_10.pkl', 'rb'))
        
    FEATURES = matrices.shape[-1]
    
    #Build Train, Dev, Test sets
    train_data = []
    train_labels = []
    dev_data = []
    dev_labels = []
    test_data = []
    test_labels = []
    test_tripId = []
    
    curTraj = ''
    assign = ''
    
    driverIds = {}
    
    for idx in range(len(keys)):
        d,t = keys[idx]
        if d in driverIds:
            dr = driverIds[d]
        else: 
            dr = len(driverIds)
            driverIds[d] = dr            
        m = matrices[idx][1:129,]
        if t != curTraj:
            curTraj = t
            r = random.random()
        if r < 0.75:
            train_data.append(m)
            train_labels.append(dr)
        elif r < 0.85:
            dev_data.append(m)
            dev_labels.append(dr)
        else:
            test_data.append(m)
            test_labels.append(dr)      
            test_tripId.append(t)

    train_data   = np.asarray(train_data, dtype="float32")
    train_labels = np.asarray(train_labels, dtype="int32")
    dev_data   = np.asarray(dev_data, dtype="float32")
    dev_labels = np.asarray(dev_labels, dtype="int32")
    test_data    = np.asarray(test_data, dtype="float32")
    test_labels  = np.asarray(test_labels, dtype="int32")
    
    rng_state = np.random.get_state()
    np.random.set_state(rng_state)
    np.random.shuffle(train_data)
    np.random.set_state(rng_state)
    np.random.shuffle(train_labels)
  
    return train_data, train_labels, dev_data, dev_labels, test_data, test_labels, test_tripId, len(driverIds), FEATURES

  
  
def convertLabelsToOneHotVector(labels, ln):
    tmp_lb = np.reshape(labels, [-1,1])
    next_batch_start = 0
    _x = np.arange(ln)
    _x = np.reshape(_x, [-1, 1])
    enc = OneHotEncoder()
    enc.fit(_x)
    labels =  enc.transform(tmp_lb).toarray()
    return labels
  
  

def returnTripLevelAccuracy(test_labels, test_tripId, probabilities, num_classes):    
    lbl = ''
    probs = []
    correct = total = 0
    for i in range(len(test_labels)):
        if lbl == test_tripId[i]:
            probs.append(probabilities[i])
        else:
            if len(probs) > 0:
                total += 1.0
                probs = np.asarray(probs)
                probs = np.mean(probs, axis=0)
                probs = (probs/np.max(probs)).astype(int)
                if np.sum(probs&test_labels[i-1].astype(int)) == 1: correct += 1
            probs = []
            probs.append(probabilities[i])
            lbl = test_tripId[i]
    if len(probs) > 0:
        total += 1.0
        probs = np.asarray(probs)
        probs = np.mean(probs, axis=0)
        probs = (probs/np.max(probs)).astype(int)      
        if np.sum(probs&test_labels[len(test_labels)-1].astype(int))==1: correct += 1
        
    return correct/total
  
 
 
if __name__ == '__main__':

    ITERATIONS = 3   # number of times to repeat the experiment
    ALL_SEG_ACC = []
    ALL_TRP_ACC = []
    
    for IT in range(0, ITERATIONS):
        tf.reset_default_graph()
        print ('\n\n************ Iteration: {} ************\n'.format(IT+1))
        
        
        # We treat images as sequences of pixel rows.
        st = time.time()
        train, train_labels, dev, dev_labels, test, test_labels, test_tripId, num_classes, FEATURES = returnTrainDevTestData()
        
        display_step = 100
        training_steps = 1000000
        batch_size = 256
        
        timesteps = 128 # Number of rows in Matrix of a Segment
        num_layers = 2 # Number of network layers
        dropouts_train = [0.0, 0.5] #dropout values for different network layers [for train]
        dropouts_dev  = [0.0, 0.0] #dropout values for different network layers [for test and dev]
        
        train_labels = convertLabelsToOneHotVector(train_labels, num_classes)  
        dev_labels = convertLabelsToOneHotVector(dev_labels, num_classes)  
        test_labels  = convertLabelsToOneHotVector(test_labels, num_classes)    
        
        print('All data is loaded in {:.1f} seconds'.format(time.time()-st))
        print('There are {} example in train, {} in dev, and {} in test set'.format(len(train), len(dev), len(test)))
        print('num_classes', num_classes)
        
        data = tf.placeholder(tf.float32, [None, 128, FEATURES])    
        target = tf.placeholder(tf.float32, [None, num_classes])
        dropout = tf.placeholder(tf.float32, [len(dropouts_train)])
        model = RNN_MODEL(data, target, dropout, num_layers)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        #print(train.shape)    
        
        train_start = time.time()
        start = time.time()
        next_batch_start = 0
        
        maxTestAccuracy = 0.0 #This will be used as a constraint to save the best model
        bestEpoch = 0    
        
        saver = tf.train.Saver() #This is the saver of the model    
        model_name = 'models/RNN_model/'        
        if os.path.exists(model_name):
            shutil.rmtree(model_name)            
        os.makedirs(model_name)
        
        steps_to_epoch = len(train)/batch_size
        
        for step in range(training_steps):
            idx_end = min(len(train),next_batch_start+batch_size)        
            sess.run(model.optimize, {data: train[next_batch_start:idx_end,:], target: train_labels[next_batch_start:idx_end,:], dropout: dropouts_train})
            
            epoch = int(step/steps_to_epoch)
            if epoch > epochs: break
            
            if epoch > bestEpoch or epoch == 0:
                acc = sess.run(model.accuracy, {data: dev, target: dev_labels, dropout: dropouts_dev})
                if epoch > 5 and acc > maxTestAccuracy:
                    maxTestAccuracy = acc
                    bestEpoch = epoch
                    save_path = saver.save(sess, model_name)
                    print('Model saved in path: {}, Accuracy: {:.2f}%, Epoch: {:d}'.format(save_path, 100*acc, epoch))
            
            if step % display_step == 0:
                loss_train = sess.run(model.cost, {data: train[next_batch_start:idx_end,:], target: train_labels[next_batch_start:idx_end,:], dropout: dropouts_dev})
                loss_dev = sess.run(model.cost, {data: dev, target: dev_labels, dropout: dropouts_dev})
                acc_train  = sess.run(model.accuracy, {data: train[next_batch_start:idx_end,:], target: train_labels[next_batch_start:idx_end,:], dropout: dropouts_dev})
                acc_dev  = sess.run(model.accuracy, {data: dev, target: dev_labels, dropout: dropouts_dev})
                print('Step {:2d}, Epoch {:2d}, Minibatch Train Loss {:.3f}, Dev Loss {:.3f}, Train-Accuracy {:.1f}%, Dev-Accuracy {:.1f}% ({:.1f} sec)'.format(step + 1, epoch, loss_train, loss_dev, 100 * acc_train, 100*acc_dev, (time.time()-start)))
                start = time.time()
            next_batch_start = next_batch_start+batch_size
            if next_batch_start >= len(train):
                rng_state = np.random.get_state()
                np.random.set_state(rng_state)
                np.random.shuffle(train)
                np.random.set_state(rng_state)
                np.random.shuffle(train_labels)
                next_batch_start = 0
        
        print("Optimization Finished!")    
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_name)
        accuracy = sess.run(model.accuracy, {data: test, target: test_labels, dropout: dropouts_dev})
        # calculate trip-level prediction accuracy
        probabilities = sess.run(model.predProbs, {data: test, target: test_labels, dropout: dropouts_dev})
        trip_level_accuracy = returnTripLevelAccuracy(test_labels, test_tripId, probabilities, num_classes)
        print('Test-Accuracy(segment): {:.2f}%, Test-Accuracy(trip): {:.2f}%,Train-Time: {:.1f}sec'.format(accuracy*100, trip_level_accuracy*100, (time.time()-train_start)))
        print('Partial Best Test-Accuracy: {:.2f}%, Best Epoch: {}'.format(maxTestAccuracy*100, bestEpoch))        
        
        ALL_SEG_ACC.append(accuracy*100)
        ALL_TRP_ACC.append(trip_level_accuracy*100)
        
    
    print ('\n\nAll Iterations are completed!')
    print ('Average Segment Accuracy: {:.2f}%, Std: {:.2f}, Min: {:.2f}, Max: {:.2f}'.format(np.mean(ALL_SEG_ACC), np.std(ALL_SEG_ACC), np.min(ALL_SEG_ACC), np.max(ALL_SEG_ACC)))
    print ('Average Trip Accuracy: {:.2f}%, Std: {:.2f}, Min: {:.2f}, Max: {:.2f}'.format(np.mean(ALL_TRP_ACC), np.std(ALL_TRP_ACC), np.min(ALL_TRP_ACC), np.max(ALL_TRP_ACC)))
