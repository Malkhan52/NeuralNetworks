# The objective is to train a multi-layered neural network to identify handwritten digits.
# We'll be using a database called MNIST which has about 6000 handwritten digit images with their labels.
# We'll use these to train a network. Then given a new image,
# the network should be able to classify it as the right digit between 0-9

# step-1: we'll download the database of images from MNIST website - this is maintained by a famous
# neural network researcher named Yann Lecun
import numpy as np
import os

def load_dataset():
    def download(filename,source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading ",filename)
        import urllib.request
        urllib.request.urlretrieve(source+filename,filename)
    # This will download the specified file from Yann Lecun's website.

    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # check if the specified file is already exist, if not it will download the file.
        with gzip.open(filename,'rb') as f:
            # open the gzip file of images
            data=np.frombuffer(f.read(),np.uint8, offset=16)
            # This is some boilerplate to extract data from the zip file
            # This data has 2 issues : its in the form of 1d array
            # We have to take this array and convert it into images
            # Each images has 28*28 pixels, it's a monochrome image i.e. only one channel

            # data is in the numpy array which we want to reshape into an array of 28*28 images
            data=data.reshape(-1,1,28,28)
            # The first dimension is the number of images, by making this -1
            # The number of images will be inferred from the value of other dimensions
            # and the length of the input array

            # The second dimension is the number of channels - here this is 1

            # The 3rd and 4th dimensions are the size of the image 28*28

            # It's in the form of bytes
            return data/np.float32(256)
        # This will convert the byte values to a float32 in the range[0,1]

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
            # Read the labels which are in a binary file again
        with gzip.open(filename,'rb') as f:
            data = np.frombuffer(f.read(),np.uint8,offset=8)
            # This gives a numpy arrayof integers, the digit value corresponding
            # to the i8mages we got earlier
        return data
    # We can now download and read the training and test datasets
    # both the images and their labels

    x_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')

    x_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    return x_train, y_train, x_test, y_test


# step 2: We'll setup a neural network with required number of layers and nodes.
# We'll also tell the network how it has to train itself
import lasagne
import theano
import theano.tensor as T

def build_NN(input_var=None):
    # We are going to create a neural network with 2 hidden layers of 800 nodes each
    # The output layer will have 10 nodes - the nodes are numbered 0-9 and the output
    # at each node will be a value between 0-1. The node with the highest value will
    # be the predicted output

    # First we have an input layer - the expected input shape is
    # 1*28*28 (for 1 image)
    # We will link this input layer to the input_var (which will be the arrat of images
    # that we'll pass in later on)

    l_in = lasagne.layers.InputLayer(shape=(None,1,28,28),input_var=input_var)

    # We will add a 20% dropout - this means that randomly 20% of the edges between the
    # inputs and the next layer will be dropped - this is done to avoid overfitting

    l_in_drop = lasagne.layers.DropoutLayer(l_in,p=0.2)

    # Add a layer with 800 nodes. Initially this will be dense/fully-connected
    # i.e. every edge possible
    # will be drawn.
    l_hid1 = lasagne.layers.DenseLayer(l_in_drop,num_units=800,
                                       nonlinearilty=lasagne.nonlinearties.rectify,
                                       W=lasagne.init.GloroUniform())

    # This layer has been initialized with some weights. There are some scheme to
    # initialize the weights so that training will be done faster, Glorot's scheme
    # is one of them

    # We will add a dropout of 50% to the hidden layer 1
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1,p=0.5)

    # Add another layer, it works exactly the same way!

    l_hid2 = lasagne.layers.DenseLayer(l_hid1_drop,num_units=800,
                                       nonlinearilty=lasagne.nonlinearties.rectify,
                                       W=lasagne.init.GloroUniform())

    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2,p=0.5)

    # Let's now add the final output layer.
    l_out = lasagne.layers.DenseLayer(l_hid2_drop, num_units=10, nonlinearity = lasagne.nonlinearities.softmax)

    # The output layer has 10 units. Softmax specifies  that each of those
    # outputs is between 0-1 and the max of the those will be the final prediction

    return l_out

    # We return the last layer, but since all the layer are linked
    # we effectively return the whole network


# We've set up the network. Now we have to tell the network how to train itself
#ie. how should it found the values of all the weights it needs to find

# We'll initialize some empty arrays wich will act as placeholders
# for the training/test data that will be given to network.

input_var = T.tensor4('inputs') # An empty 4 dimensional array
target_var = T.ivector('targets') # An empty 1 dimensional integer array to represent the labels

network = build_NN(input_var) #Call the function that initializes the neural network

# In trainig we are going to follow the steps below
# a. compute an error function
prediction = lasagne.objectives.categorical_crossentropy(prediction, target_var)
# Categorical cross entropy is one of the standard error functions with classification problems
loss = loss.mean()

# b. We'll the network how to updates it's all weights based on the value of the error function
params = lasagne.layers.get_all_params(network, trainable=True)
# Current value of all the weight in a trainig step.
# This is based on stochastic Gradient Descent - the idea is simple
# Find the slope of the error function at the current point and move downwards in the direction of that slope

# We'll use theano to compile a function that is going to represent a single training step
# ie. compute the error, find the current weights and update the weights
train_fn = theano.function([input_var, target_var], loss, updates=updates)
# calling this function for a certain number of times will train the neural network

# step 3: We'll feed the training data to neural network
num_training_steps = 10

for step in range(num_training_steps):
    train_err = train_fn(x_train,y_train)
    print("Current step is "+ str(step))

# step 4: We'll check how the output for one image

# step 5: We'll feed a test data set of 10000 images to trained neural network and check it's accuracy
