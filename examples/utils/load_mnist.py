import gzip
import os
import urllib.request as request
from os import path

import numpy as np

MNIST_FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
               "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]


def download(url, file_path):
    # Creates the directory path for further downloading
    dir_path = path.dirname(file_path)

    # Checks if path exists
    if not path.exists(dir_path):
        # If not, create its directory
        os.makedirs(dir_path)

    # Retrieves the file
    request.urlretrieve(url, file_path)


def download_mnist(file_path):
    # URL for MNIST dataset
    url = 'http://yann.lecun.com/exdb/mnist/'

    # For each file
    for mnist in MNIST_FILES:
        # We create a path to download the file
        mnist_path = os.path.join(file_path, mnist)

        # If the path does not exists
        if not path.exists(mnist_path):
            # Downloads the file
            download(url + mnist, mnist_path)


def load_single_mnist(mnist_dir, mnist_file, bits, shape):
    # Trying to open desired file
    with gzip.open(os.path.join(mnist_dir, mnist_file)) as fd:
        # Reading to buffer
        buf = fd.read()

        # From buffer, we actually load the file
        loaded = np.frombuffer(buf, dtype=np.uint8)

        # Reshaping the data
        data = loaded[bits:].reshape(shape)

        return data


def load_mnist():
    # Directory to MNIST dataset
    mnist_dir = 'datasets/mnist/'

    # If there is no directory
    if not path.exists(mnist_dir):
        # Downloads the dataset
        download_mnist(mnist_dir)

    # If there is a directory
    else:
        # Check if files have been downloaded
        exists = [path.exists(os.path.join(mnist_dir, f)) for f in MNIST_FILES]

        # If they have not been downloaded
        if not np.all(exists):
            # Downloads the dataset
            download_mnist(mnist_dir)

    # Loading training samples
    X_train = load_single_mnist(
        mnist_dir, 'train-images-idx3-ubyte.gz', 16, (60000, 28 * 28)).astype(float)

    # Loading training labels
    Y_train = load_single_mnist(
        mnist_dir, 'train-labels-idx1-ubyte.gz', 8, (60000))

    # Loading validation samples
    X_val = load_single_mnist(
        mnist_dir, 't10k-images-idx3-ubyte.gz', 16, (10000, 28 * 28)).astype(float)

    # Loading validation labels
    Y_val = load_single_mnist(
        mnist_dir, 't10k-labels-idx1-ubyte.gz', 8, (10000))

    # Normalizing samples
    X_train /= 255.
    X_val /= 255.

    return X_train, Y_train, X_val, Y_val
