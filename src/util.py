import struct
import numpy as np
from array import array
from os.path import join
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import seaborn as sns


def train_test_split(X, y, split_size=0.3):
    '''
        Split input(X) and output(y) into training and testing portions
        Arguments:
        X: array of input
        y: array of output
        split_size: float between 0.0 and 1.0 representing percentage of
                    data to be used in training

        Returns:
        X_train, X_test, y_train, y_test
    '''

    # get number of samples
    S = len(X)

    # case of split size being integer greater than one
    if split_size>1:

        # if split size is non-integer greater than one raise error
        if not isinstance(split_size, int):
            raise TypeError("split size must be integer if greater than one and "
                            "should represent the number of samples in the "
                            "training")
        
        # if split size greater than number of samples raise error
        if split_size>S:
            raise ValueError("if split size is integer greater than one then "
                                "it cannot be greater than the size of X and y")
        S_train = split_size

    # case of split size being float between 0.0 and 1.0
    else:
        S_train = round(S*split_size)

    # randomly select training and testing indices
    indices = np.random.permutation(S)
    train_indices = indices[:S_train]
    test_indices = indices[S_train:]

    # training data
    X_train = X[train_indices]
    y_train = y[train_indices]

    # testing data
    X_test = X[test_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


def confusion_matrix_plot(y_true, y_pred, figsize=(8,6), out_file="confusion_matrix.png"):
    ''' Plot confusion matrix '''
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('MNIST Confusion Matrix')
    plt.savefig(out_file)   


def plot_random_mnist(images, y_true, y_pred, num_images=9, out_file="mnist_failed_images.png"):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    incorrect_indices = np.where(y_pred != y_true)[0]
    np.random.shuffle(incorrect_indices)
    incorrect_indices = incorrect_indices[:num_images]

    X_img = [images[i] for i in incorrect_indices]
    titles = [f"Predicted {y_pred[i]}     Actual {y_true[i]}" for i in incorrect_indices]

    show_mnist_images(X_img, titles, out_file=out_file)

#
# The following is code from https://www.kaggle.com/code/hojjatk/read-mnist-dataset
# which includes functions for loading the idx-ubyte files of MNIST dataset as well
# as plotting the images
#

#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)


def show_mnist_images(images, title_texts, out_file="mnist_images.png"):
    cols = 3
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(17,17))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);
        index += 1

    plt.savefig(out_file)



