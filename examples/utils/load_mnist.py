import gzip
import os
import urllib.request as request
from os import path

import numpy as np

FILES_MNIST = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
               "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]


def download(url, file_name):
    # Creates the directory path for further downloading
    dir_name = path.dirname(file_name)

    # Checks if the path exists
    if not path.exists(dir_name):
        # If not, create its directory
        os.makedirs(dir_name)

    # Retrieves the file
    request.urlretrieve(url, file_name)


def download_mnist(file_name):
    # URL for MNIST dataset
    url = 'http://yann.lecun.com/exdb/mnist/'

    # For each possible file
    for mnist in FILES_MNIST:
        # We create a path to download the file
        mnist_path = os.path.join(file_name, mnist)

        # If the path does not exists
        if not path.exists(mnist_path):
            # Downloads the file
            download(url + mnist, mnist_path)


def load_single_mnist(dir_mnist, file_mnist, bits, shape):
    # Trying to open desired file
    with gzip.open(os.path.join(dir_mnist, file_mnist)) as fd:
        # Reading to buffer
        buf = fd.read()

        # From buffer, we actually load the file
        loaded = np.frombuffer(buf, dtype=np.uint8)

        # Reshaping the data
        data = loaded[bits:].reshape(shape)

        return data


def load_mnist():
    # Directory to MNIST dataset
    dir_mnist = 'datasets/mnist/'

    # If there is no directory
    if not path.exists(dir_mnist):
        # Downloads the dataset
        download_mnist(dir_mnist)

    # If there is a directory
    else:
        # Check if files have been downloaded
        exists = [path.exists(os.path.join(dir_mnist, f)) for f in FILES_MNIST]

        # If they have not been downloaded
        if not np.all(exists):
            # Downloads the dataset
            download_mnist(dir_mnist)

    # Loading training samples
    X_train = load_single_mnist(
        dir_mnist, 'train-images-idx3-ubyte.gz', 16, (60000, 28 * 28)).astype(float)

    # Loading training labels
    Y_train = load_single_mnist(
        dir_mnist, 'train-labels-idx1-ubyte.gz', 8, (60000))

    # Loading validation samples
    X_val = load_single_mnist(
        dir_mnist, 't10k-images-idx3-ubyte.gz', 16, (10000, 28 * 28)).astype(float)

    # Loading validation labels
    Y_val = load_single_mnist(
        dir_mnist, 't10k-labels-idx1-ubyte.gz', 8, (10000))

    # Normalizing samples
    X_train /= 255.
    X_val /= 255.

    return X_train, Y_train, X_val, Y_val
