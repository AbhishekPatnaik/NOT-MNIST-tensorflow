from __future__ import print_function
import numpy as np
import tensorflow as tf
from  six.moves import cPickle as pickle


pickle_file = 'notMNIST.pickle'

with open(pickle_file,'rb') as f:
    save = pickle.load(f)
    train_datasets = save['train_dataset']
    train_labels = save['train_labels']
    valid_datasets = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_datasets = save['test_dataset']
    test_labels = save['test_labels']

print(train_datasets)
image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_datasets, train_labels = reformat(train_datasets, train_labels)
valid_datasets, valid_labels = reformat(valid_datasets, valid_labels)
test_dataset, test_labels = reformat(test_datasets, test_labels)
print('Training set', train_datasets.shape, train_labels.shape)
print('Validation set', valid_datasets.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

#this is to expedate the process

train_subset = 10000
beta = 0.01
graph = tf.Graph()

with graph.as_default():
    # Input data
    #they are all constants
    tf_train_dataset = tf.constant(train_datasets[:train_subset,:])
    tf_train_labels = tf.constant(train_labels[:train_subset])
    tf_valid_dataset = tf.constant(valid_datasets)
    tf_test_dataset = tf.constant(test_dataset)

    #variables
    #these are varibales we want to update and optimize
    weights = tf.Variable(tf.truncated_normal([image_size*image_size,num_labels]))
    baises = tf.Variable(tf.zeros([num_labels]))
    #Training computation
    logits = tf.matmul(tf_train_dataset,weights)+baises
    #Original loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels,logits=logits))

    #l2 reguralisation
    l2  = tf.nn.l2_loss(weights)
    loss = tf.reduce_mean(loss + beta * l2)

    #Optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    #prediction
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + baises)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + baises)

    num_steps = 1000



def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
                / predictions.shape[0])

with tf.Session(graph=graph) as session:
    # This is a one-time operation which ensures the parameters get initialized as
    # we described in the graph: random weights for the matrix, zeros for the
    # biases.
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
    # Run the computations. We tell .run() that we want to run the optimizer,
    # and get the loss value and the training predictions returned as numpy
    # arrays
        _,l,prediction = session.run([optimizer,loss,train_prediction])
        if step%100 == 0:
            print('loss at step{}: {}'.format(step,l))
            print('Training accuracy: {:.1f}'.format(accuracy(prediction,train_labels[:train_subset, :])))
            # Calling .eval() on valid_prediction is basically like calling run(), but
            # just to get that one numpy array. Note that it recomputes all its graph
            # dependencies.

            # You don't have to do .eval above because we already ran the session for the
            # train_prediction
            print('Validation accuracy: {:.1f}'.format(accuracy(valid_prediction.eval(),
                                                                valid_labels)))
    print('Test accuracy: {:.1f}'.format(accuracy(test_prediction.eval(), test_labels)))

# Neural networks with 1 hidden layer L2 Reguralization Relu

num_nodes = 1024
batch_size = 128
beta = 0.01

graph = tf.Graph()
graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_datasets)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    weights_1 = tf.Variable(tf.truncated_normal([image_size * image_size, num_nodes]))
    biases_1 = tf.Variable(tf.zeros([num_nodes]))
    weights_2 = tf.Variable(tf.truncated_normal([num_nodes, num_labels]))
    biases_2 = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    logits_1 = tf.matmul(tf_train_dataset, weights_1) + biases_1
    relu_layer = tf.nn.relu(logits_1)
    logits_2 = tf.matmul(relu_layer, weights_2) + biases_2
    # Normal loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_2,labels= tf_train_labels))
    # Loss function with L2 Regularization with beta=0.01
    regularizers = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2)
    loss = tf.reduce_mean(loss + beta * regularizers)

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training
    train_prediction = tf.nn.softmax(logits_2)

    # Predictions for validation
    logits_1 = tf.matmul(tf_valid_dataset, weights_1) + biases_1
    relu_layer = tf.nn.relu(logits_1)
    logits_2 = tf.matmul(relu_layer, weights_2) + biases_2

    valid_prediction = tf.nn.softmax(logits_2)

    # Predictions for test
    logits_1 = tf.matmul(tf_test_dataset, weights_1) + biases_1
    relu_layer = tf.nn.relu(logits_1)
    logits_2 = tf.matmul(relu_layer, weights_2) + biases_2

    test_prediction = tf.nn.softmax(logits_2)
num_steps = 3001

with tf.Session(graph=graph) as session:
    #Varifies that the variables are initialised initially
    tf.initialize_all_variables().run()
    print('Initialized')
    for steps in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
            offset = (steps*batch_size) % (train_labels.shape[0]-batch_size)
            #Generate a minibatch
            batch_data = train_datasets[offset:(offset+batch_size),:]
            batch_labels = train_labels[offset:(offset+batch_size),:]
            # Prepare a dictionary telling the session where to feed the data
            # the key of the dictionary is the placeholder node of the graph to be fed
            # and the value is the numpy array to be fed
            feed_dict = {tf_train_dataset:batch_data,tf_train_labels:batch_labels}
            _ ,l , prediction = session.run([optimizer,loss,train_prediction],feed_dict=feed_dict)
            if steps%500 ==0:
                print('Minibatch loss at step {}: {}'.format(steps,l))
                print('Minibatch accuracy: {:.1f} '.format(accuracy(prediction,batch_labels)))
                print('Validation accuracy: {:.1f}'.format(accuracy(valid_prediction.eval(),valid_labels)))
    print("Test accuracy: {:.1f}".format(accuracy(test_prediction.eval(), test_labels)))

#Lets see extreme case of overfitting .Reduce the your training data to a few batches.lets see what happens??


train_datasets_2 = train_datasets[:500,:]
train_labels_2 = train_labels[:500,:]



with tf.Session(graph=graph) as session:
    #Varifies that the variables are initialised initially
    tf.initialize_all_variables().run()
    print('Initialized')
    for steps in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
            offset = (steps*batch_size) % (train_labels_2.shape[0]-batch_size)
            #Generate a minibatch
            batch_data = train_datasets_2[offset:(offset+batch_size),:]
            batch_labels = train_labels_2[offset:(offset+batch_size),:]
            # Prepare a dictionary telling the session where to feed the data
            # the key of the dictionary is the placeholder node of the graph to be fed
            # and the value is the numpy array to be fed
            feed_dict = {tf_train_dataset:batch_data,tf_train_labels:batch_labels}
            _ ,l , prediction = session.run([optimizer,loss,train_prediction],feed_dict=feed_dict)
            if steps%500 ==0:
                print('Minibatch loss at step {}: {}'.format(steps,l))
                print('Minibatch accuracy: {:.1f} '.format(accuracy(prediction,batch_labels)))
                print('Validation accuracy: {:.1f}'.format(accuracy(valid_prediction.eval(),valid_labels)))
    print("Test accuracy: {:.1f}".format(accuracy(test_prediction.eval(), test_labels)))


# Introduce dropout on the hidden layer of the neural network. Remember : Dropout should only be applied during the
# training and not testing stage, otherwise your evaluation results would be stochastic as well.
# TensorFlow provides nn.dropout() for that, but you have to make sure it's only inserted during training.

num_nodes = 1024
batch_size = 128
beta = 0.01

graph = tf.Graph()
with tf.Session(graph=graph) as session:
    # Input data . For the training data we use a placeholder that will be fed at the run time.with a training
    # minibatch
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_datasets)
    tf_test_dataset = tf.constant(test_dataset)

    #Input Variables
    weights_1 = tf.Variable(tf.truncated_normal([image_size*image_size,num_labels]))
    baises_1 = tf.Variable(tf.zeros([num_nodes]))
    weights_2 = tf.Variable(tf.truncated_normal([num_nodes, num_labels]))
    baises_2 = tf.Variable(tf.zeros([num_labels]))

    #Training computation
    logits_1 = tf.matmul(tf_train_dataset,weights_1) + baises_1
    # relu_layer
    relu_layer = tf.nn.relu(logits_1)
    # Dropout on hidden layer :Relu layer
    keep_prob = tf.placeholder("float")
    relu_layer_dropout = tf.nn.dropout(relu_layer, keep_prob)

    logits_2 = tf.matmul(relu_layer_dropout,weights_2)+baises_2
    #Normal loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_2,labels=tf_train_labels))

    #loss fucntion with L2 reguralisation with beta = 0.01
    regularizers = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2)
    loss = tf.reduce_mean(loss + beta * regularizers)

    #Optimizer

    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    #Prediction
    train_prediction = tf.nn.softmax(logits_2)

    #Prediction for validation
    logits_1 = tf.matmul(tf_valid_dataset,weights_1)+baises_1
    relu_layer =  tf.nn.relu(logits_1)
    logits_2 = tf.matmul(relu_layer,weights_2) + baises_2

    valid_prediction = tf.nn.softmax(logits_2)

    # Prediction for testing datasets
    logits_1 = tf.matmul(tf_test_dataset, weights_1) + biases_1
    relu_layer = tf.nn.relu(logits_1)
    logits_2 = tf.matmul(relu_layer, weights_2) + biases_2

    test_prediction = tf.nn.softmax(logits_2)

num_steps = 3001

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialised')
    #creating Minibatch
    for steps in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (steps * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate Minibatch
        batch_data = train_datasets[offset:(offset+batch_size),:]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, keep_prob: 0.5}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print("Minibatch loss at step {}: {}".format(step, l))
            print("Minibatch accuracy: {:.1f}".format(accuracy(predictions, batch_labels)))
            print("Validation accuracy: {:.1f}".format(accuracy(valid_prediction.eval(), valid_labels)))
    print("Test accuracy: {:.1f}".format(accuracy(test_prediction.eval(), test_labels)))





































































































