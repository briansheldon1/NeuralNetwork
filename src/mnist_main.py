import os
import numpy as np
from network import Network
from util import train_test_split, MnistDataloader, show_mnist_images, \
                 confusion_matrix_plot, plot_random_mnist


from sklearn.metrics import confusion_matrix
import seaborn as sns

curr_dir =  os.path.dirname(os.path.abspath(__file__))

def mnist_main(hidden_layers=(32, 16), alpha=0.001, gamma=0.9, epochs=50,
               load_from_file = None, save_to_file = None, verbose=True, 
               plot_confmat = True):
    
    # load data
    train_images_path = os.path.join(curr_dir, '../mnist_data/train-images.idx3-ubyte')
    train_labels_path = os.path.join(curr_dir, '../mnist_data/train-labels.idx1-ubyte')

    test_images_path = os.path.join(curr_dir, '../mnist_data/t10k-images.idx3-ubyte')
    test_labels_path = os.path.join(curr_dir, '../mnist_data/t10k-labels.idx1-ubyte')

    data_loader = MnistDataloader(train_images_path, train_labels_path, test_images_path, test_labels_path)
    (X_train_pre, y_train_pre), (X_test_pre, y_test_pre) = data_loader.load_data()
    
    # Normalize data (flatten images and divide by 255, vectorize labels)
    X_train = np.array([np.array(x).flatten()/255.0 for x in X_train_pre])
    X_test = np.array([np.array(x).flatten()/255.0 for x in X_test_pre])

    y_train = np.eye(10)[y_train_pre]
    y_test = np.eye(10)[y_test_pre]


    # define nn
    layers = (X_train.shape[1], *hidden_layers, y_train.shape[1])
    nn = Network(layers, alpha=alpha, gamma=gamma)
    

    # load or train model
    if load_from_file is not None:
        nn.load_model(os.path.join(curr_dir, load_from_file))
    else:
        mnist_history = nn.fit(X_train, y_train, verbose=verbose, epochs=epochs, batch_size=200)

        if save_to_file is not None:
            nn.save_model(os.path.join(curr_dir, save_to_file))


    # test model accuracy
    mnist_accuracy = nn.test_accuracy(X_test, y_test, model_type='categorical')
    print(f"MNIST Test Final Accuracy: {round(mnist_accuracy*100, 4)}% \n")


    # get predicted labels for analysis
    y_pred = nn.predict(X_test)
    y_pred = [np.argmax(pred) for pred in y_pred]


    # plot confusion matrix
    if plot_confmat:
        confusion_matrix_plot(y_test_pre, y_pred, 
                            out_file=os.path.join(curr_dir, '../plots/mnist_confmatrix.png'))

    plot_random_mnist(X_test_pre, y_test_pre, y_pred, 
                      out_file=os.path.join(curr_dir, '../plots/mnist_failed_images.png'))



if __name__ == '__main__':
    load_from_file = '../models/mnist_model.pkl'
    mnist_main(load_from_file=load_from_file, plot_confmat=True)

