'''
                                Import packages
'''

import pickle
import DNN
import Plots_Module as plts
import gzip

from numpy import zeros, reshape


'''
                                Parameters
''' 

epochs = 100
lr = 0.1
batch_length=10
strOptim = 'RMSprop'
gamma = 0.9
beta_1 = 0.9
beta_2 = 0.999
rho = 0.9


'''
                                One hot encoding
'''

def vectorized_result(j):
    e = zeros((10, 1))
    e[j] = 1.0
    return e


'''
                               Split the train, test and validation sets     
'''

def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    
    ###
    training_inputs = [reshape(x, (784, 1)) for x in tr_d[0]]
    
    training_results = tr_d[1]
    training_results_one_hot = [vectorized_result(y) for y in tr_d[1]]
    
    training_data = list(zip(training_inputs, training_results))
    training_data_one_hot = list(zip(training_inputs, training_results_one_hot))


    ###
    validation_inputs = [reshape(x, (784, 1)) for x in va_d[0]]
    
    validation_results = va_d[1]
    validation_results_one_hot = [vectorized_result(y) for y in va_d[1]]
    
    validation_data = list(zip(validation_inputs, validation_results))
    validation_data_one_hot = list(zip(validation_inputs, validation_results_one_hot))
    
    
    ###
    test_inputs = [reshape(x, (784, 1)) for x in te_d[0]]
    
    test_results = te_d[1]
    test_results_one_hot = [vectorized_result(y) for y in te_d[1]]
    
    test_data = list(zip(test_inputs, test_results))
    test_data_one_hot = list(zip(test_inputs, test_results_one_hot))
    
    
    
    return training_data, training_data_one_hot, validation_data, validation_data_one_hot, test_data, test_data_one_hot



training_data, training_data_one_hot, validation_data, validation_data_one_hot, test_data, test_data_one_hot = load_data_wrapper()


'''
                                    Neural network training
'''

metrics = ['train_acc', 'test_acc', 'train_loss', 'test_loss']
MNIST_DNN = DNN.neural_network([784, 30, 10])
train_acc, val_acc, train_loss, val_loss = MNIST_DNN.train_network(training_data_one_hot=training_data_one_hot, training_data=training_data, 
                                                                   test_data_one_hot=test_data_one_hot, test_data=test_data, 
                                                                   mini_batch_size=batch_length, strOptim=strOptim,
                                                                   epochs=epochs, learning_rate=lr, metrics=metrics, gamma=gamma)


plts.plot_accuracy(number_of_epochs=epochs, train_acc=train_acc, val_acc=val_acc)
plts.plot_loss(number_of_epochs=epochs, train_loss=train_loss, val_loss=val_loss)
