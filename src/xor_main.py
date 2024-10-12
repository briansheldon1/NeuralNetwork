import numpy as np
from network import Network
import os
from util import train_test_split

### This file is for testing the model with just XOR data ###


def xor_main(layers=(2,8,1), alpha=0.1, gamma=0.9, 
             load_from_file = None, save_to_file = None):
    ''' 
        Main function to test XOR data using neural network 
        Arguments:
            layers: tuple or array of layer sizes (includes non-hidden layers)
            alpha: learning rate
            gamma: momentum constant (using nesterov's momentum)
            load_from_file: file to load model from (skips training)
            save_to_file: file to save model to (if None does not save)
    '''

    # create nn
    nn = Network(layers, alpha=alpha, gamma=gamma)

    # get xor data
    X = np.random.randint(2, size=(10000,2))
    y = np.array([1 if x[0]!=x[1] else 0 for x in X])

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # if loading from file then load data from file
    curr_dir =  os.path.dirname(os.path.abspath(__file__))
    if load_from_file is not None:
        nn.load_model(os.path.join(curr_dir, load_from_file))
    else:
        nn.fit(X_train, y_train, verbose=True, epochs=100, batch_size=200)

        if save_to_file is not None:
            nn.save_model(os.path.join(curr_dir, save_to_file))

    # test accuracy
    accuracy = nn.test_accuracy(X_test, y_test)
    print(f"\nFinal Accuracy: {accuracy*100}%")

    
if __name__ == '__main__':
    xor_main()