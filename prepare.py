


import pickle
import gzip


import numpy as np

def load_data():

    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():

    tr_d, va_d, te_d = load_data()
    #print(tr_d[0].T.shape)
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_results=np.asarray(training_results).T
    training_data = tr_d[0].T
    validation_results = [vectorized_result(y) for y in va_d[1]]
    validation_results=np.asarray(validation_results).T
    validation_data = va_d[0].T
    testing_results = [vectorized_result(y) for y in te_d[1]]
    testing_results=np.asarray(testing_results).T
    test_data = te_d[0].T
    return (training_data,training_results, validation_data,validation_results, test_data,testing_results)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = [0]*10
    e[j] = 1.0
    return e
