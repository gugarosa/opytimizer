import os
import sys
from ctypes import *

import numpy as np


class LibOPF:
    """A class to hold the LibOPF's integration.

    """

    def __init__(self):
        """Initialization method.

        """

        # Creates the OPF property
        self._OPF = CDLL(os.environ['OPF_DIR']+'/OPF.so')

    def wrap_function(lib, funcname, restype, argtypes):
        """Wraps a function using ctypes.

        Args:
            lib (func): The function pointer to be loaded.
            funcname (string): The name of the function.
            restype (types): Function's return type.
            argtypes (types): Types of the arguments.

        """

        # Gets the function object
        func = lib.__getattr__(funcname)

        # Gets the type of the response
        func.restype = restype

        # Gets the arguments' types
        func.argtypes = argtypes

        return func


class Set(Structure):
    """A class to hold the Set's structure.

    """

    # Fields that belongs to the structure
    _fields_ = [
        ("elems", c_int),
        ("next", POINTER(Set))
    ]


class SNode(Structure):
    """A class to hold the subgraph's Node structure.

    """

    # Fields that belongs to the structure
    _fields_ = [
        ("pathvalue", c_float),
        ("dens", c_float),
        ("radius", c_float),
        ("label", c_int),
        ("root", c_int),
        ("pred", c_int),
        ("truelabel", c_int),
        ("position", c_int),
        ("feat", POINTER(c_float)),
        ("status", c_char),
        ("relevant", c_char),
        ("nplatadj", c_int),
        ("adj", POINTER(Set))
    ]


class Subgraph(Structure, LibOPF):
    """A class to hold the Subgraph structure.

    """

    # Fields that belongs to the structure
    _fields_ = [
        ("node", POINTER(SNode)),
        ("nnodes", c_int),
        ("nfeats", c_int),
        ("bestk", c_int),
        ("nlabels", c_int),
        ("df", c_float),
        ("mindens", c_float),
        ("maxdens", c_float),
        ("k", c_float),
        ("ordered_list_of_nodes", POINTER(c_int))
    ]

    def __init__(self):
        """Initialization method.

        """

        # Override its parent class
        super().__init__()


class OPF(LibOPF):
    """Wraps methods from the LibOPF.

    """

    def __init__(self):
        """Initialization method.

        """

        # Overrides its parent class
        super().__init__()

    def _readsubgraph(self, dataset):
        """Reads a subgraph from a .opf file.

        Args:
            dataset (string): Path to .opf dataset file.

        """

        print('Reading data file ...')

        # Creates the pointer to the function
        readsubgraph = self._OPF.ReadSubgraph

        # Gets the type of the response
        readsubgraph.restype = POINTER(Subgraph)

        # Gets the arguments types
        readsubgraph.argtypes = [c_char_p]

        # Actually uses the function
        g = readsubgraph(dataset)

        return g

    def _writesubgraph(self, subgraph, file_name):
        """Writes a subgraph to a .opf file.

        Args:
            subgraph (Subgraph): Subgraph object to be written.
            file_name (string): Path to the file that will be saved.

        """

        print('Writing data file ...')

        # Creates the pointer to the function
        writesubgraph = self._OPF.WriteSubgraph

        # Gets the arguments types
        writesubgraph.argtypes = [POINTER(Subgraph), c_char_p]

        # Actually uses the function
        writesubgraph(subgraph, file_name)

    def _destroysubgraph(self, subgraph):
        """Destroys a subgraph.

        Args:
            subgraph (Subgraph): Subgraph object to be destroyed.

        """

        print('Deallocating memory ...')

        # Creates the pointer to the function
        destroysubgraph = self._OPF.DestroySubgraph

        # Gets the arguments types
        destroysubgraph.argtypes = [POINTER(POINTER(Subgraph))]

        # Actually uses the function
        destroysubgraph(subgraph)

    def _writemodelfile(self, subgraph, file_name):
        """Writes a subgraph to a model file.

        Args:
            subgraph (Subgraph): Subgraph object to be written.
            file_name (string): Path to the file that will be saved.

        """

        print('Writing classifier\'s model file ...')

        # Creates the pointer to the function
        writemodelfile = self._OPF.opf_WriteModelFile

        # Gets the arguments types
        writemodelfile.argtypes = [POINTER(Subgraph), c_char_p]

        # Actually uses the function
        writemodelfile(subgraph, file_name)

    def _readmodelfile(self, file_name):
        """Reads a model file to a subgraph.

        Args:
            file_name (string): Path to the model file that will be read.

        """

        print('Reading classifier\'s model file ...')

        # Creates the pointer to the function
        readmodelfile = self._OPF.opf_ReadModelFile

        # Gets the type of the response
        readmodelfile.restype = POINTER(Subgraph)

        # Gets the arguments types
        readmodelfile.argtypes = [c_char_p]

        # Actually uses the function
        g = readmodelfile(file_name)

        return g

    def _modelfile2txt(self):
        """Converts the classifier.opf from binary to text.

        """

        print('Converting classifier.opf from binary to text ...')

        # Creates the pointer to the function
        modelfile2txt = self._OPF.opf_ModelFile2Txt

        # Actually uses the function
        modelfile2txt()

    def _writeoutputfile(self, subgraph, file_name):
        """Writes an output file.

        Args:
            subgraph (Subgraph): Subgraph object to be written.
            file_name (string): Path to the file that will be saved.

        """

        print('Writing output file ...')

        # Creates the pointer to the function
        writeoutputfile = self._OPF.opf_WriteOutputFile

        # Gets the argument types
        writeoutputfile.argtypes = [POINTER(Subgraph), c_char_p]

        # Actually uses the function
        writeoutputfile(subgraph, file_name)

    def _readoutputfile(self, subgraph, file_name):
        """Reads an output file.

        Args:
            subgraph (Subgraph): Subgraph object to be read.
            file_name (string): Path to the file that will be read.

        """

        print('Reading output file ...')

        # Creates the pointer to the function
        readoutputfile = self._OPF.opf_ReadOutputFile

        # Gets the argument types
        readoutputfile.argtypes = [POINTER(Subgraph), c_char_p]

        # Actually uses the function
        readoutputfile(subgraph, file_name)

    def _training(self, train):
        """Trains a model using supervised OPF.

        Args:
            train (Subgraph): Training subgraph.

        """

        print('Training with supervised OPF ...')

        # Creates the pointer to the function
        training = self._OPF.opf_OPFTraining

        # Gets the argument types
        training.argtypes = [POINTER(Subgraph)]

        # Actually uses the function
        training(train)

    def _classifying(self, train, test):
        """Classifies a model.

        Args:
            train (Subgraph): Training subgraph.
            test (Subgraph): Test subgraph.

        """

        print('Classifying data ...')

        # Creates the pointer to the function
        classifying = self._OPF.opf_OPFClassifying

        # Gets the argument types
        classifying.argtypes = [POINTER(Subgraph)]

        # Actually uses the function
        classifying(test)

    def _bestkmincut(self, train, k_min, k_max):
        """Performs the best subgraph minimum cut.

        Args:
            train (Subgraph): Training subgraph.
            k_min (int): Minimum cut value.
            k_max (int): Maximum cut value.

        """

        print('Estimating the best k  by minimum cut ...')

        # Creates the pointer to the function
        bestkmincut = self._OPF.opf_BestkMinCut

        # Gets the argument types
        bestkmincut.argtypes = [POINTER(Subgraph), c_int, c_int]

        # Actually uses the function
        bestkmincut(train, k_min, k_max)

    def _clustering(self, train):
        """Clusters a model using OPF clustering.

        Args:
            train (Subgraph): Training subgraph.

        """

        print('Clustering with OPF ...')

        # Creates the pointer to the function
        clustering = self._OPF.opf_OPFClustering

        # Gets the argument types
        clustering.argtypes = [POINTER(Subgraph)]

        # Actually uses the function
        clustering(train)

    def _knn_classify(self, train, test):
        """Classifies a model using KNN.

        Args:
            train (Subgraph): Training subgraph.
            test (Subgraph): Test subgraph.

        """

        print('Classifying with KNN ...')

        # Creates the pointer to the function
        knn_classify = self._OPF.opf_OPFknnClassify

        # Gets the argument types
        knn_classify.argtypes = [POINTER(Subgraph), POINTER(Subgraph)]

        # Actually uses the function
        knn_classify(train, test)

    def _elimmaxbelowH(self, subgraph, h):
        """Eliminates the maximum height in the subgraph.

        Args:
            subgraph (Subgraph): Subgraph to reduce its height.
            h (float): Maximum height.

        """

        print('Reducing by maximum height ...')

        # Creates the pointer to the function
        elimmaxbelowH = self._OPF.opf_ElimMaxBelowH

        # Gets the argument types
        elimmaxbelowH.argtypes = [POINTER(Subgraph), c_float]

        # Actually uses the function
        elimmaxbelowH(subgraph, h)

    def _elimmaxbelowA(self, subgraph, a):
        """Eliminates the maximum area in the subgraph.

        Args:
            subgraph (Subgraph): Subgraph to reduce its area.
            a (float): Maximum area.

        """

        print('Reducing by maximum area ...')

        # Creates the pointer to the function
        elimmaxbelowA = self._OPF.opf_ElimMaxBelowArea

        # Gets the argument types
        elimmaxbelowA.argtypes = [POINTER(Subgraph), c_int]

        # Actually uses the function
        elimmaxbelowA(subgraph, a)

    def _elimmaxbelowV(self, subgraph, v):
        """Eliminates the maximum volume in the subgraph.

        Args:
            subgraph (Subgraph): Subgraph to reduce its volume.
            v (float): Maximum volume.

        """

        print('Reducing by maximum volume ...')

        # Creates the pointer to the function
        elimmaxbelowV = self._OPF.opf_ElimMaxBelowVolume

        # Gets the argument types
        elimmaxbelowV.argtypes = [POINTER(Subgraph), c_int]

        # Actually uses the function
        elimmaxbelowV(subgraph, v)

    def _accuracy(self, subgraph):
        """Computes the model's accuracy.

        Args:
            subgraph (Subgraph): Subgraph to compute its accuracy.

        """

        print('Calculating accuracy ...')

        # Creates the pointer to the function
        accuracy = self._OPF.opf_Accuracy

        # Gets the type of the response
        accuracy.restype = c_float

        # Gets the argument types
        accuracy.argtypes = [POINTER(Subgraph)]

        # Actually uses the function
        result = accuracy(subgraph)

        return result


def dome_height(opf, subgraph, value):
    """Performing the subgraph reduction by height.

    Args:
        opf (OPF): OPF class instance.
        subgraph (Subgraph): Subgraph to be reduced.
        value (float): Maximum value.

    """

    # Defines the maximum height
    Hmax = 0.0

    # Iterate through all nodes
    for i in range(subgraph.contents.nnodes):
        # If node's density is bigger than maximum height
        if (subgraph.contents.node[i].dens > Hmax):
            # Apply maximum height as its density
            Hmax = subgraph.contents.node[i].dens

    # Performs the reduction
    opf._elimmaxbelowH(subgraph, (Hmax * value))


def dome_area(opf, subgraph, value):
    """Performing the subgraph reduction by area.

    Args:
        opf (OPF): OPF class instance.
        subgraph (Subgraph): Subgraph to be reduced.
        value (float): Maximum value.

    """

    # Performs the reduction
    opf._elimmaxbelowA(subgraph, int(value * subgraph.contents.nnodes))


def dome_volume(opf, subgraph, value):
    """Performing the subgraph reduction by volume.

    Args:
        opf (OPF): OPF class instance.
        subgraph (Subgraph): Subgraph to be reduced.
        value (float): Maximum value.

    """

    # Defines the maximum volume
    Vmax = 0.0

    # Iterate through all nodes
    for i in range(subgraph.contents.nnodes):
        # Sum all the volumes
        Vmax += subgraph.contents.node[i].dens

    # Performs the reduction
    opf._elimmaxbelowH(subgraph, int(
        value * (Vmax / subgraph.contents.nnodes)))


def eliminate(op, opf, subgraph, value):
    """Performing the subgraph reduction.

    Args:
        op (int): Operation to be chosen.
        opf (OPF): OPF class instance.
        subgraph (Subgraph): Subgraph to be reduced.
        value (float): Maximum value.

    """

    # Creates a switcher of operations
    switcher = {
        0: lambda: dome_height(opf, subgraph, value),
        1: lambda: dome_area(opf, subgraph, value),
        2: lambda: dome_volume(opf, subgraph, value),
    }

    # Returns the switcher based on chosen operation
    return switcher.get(op, lambda: "ERROR: option invalid")()


def _cluster(opf, train_file, op, value):
    """Performs the OPF clustering.

    Args:
        opf (OPF): OPF class instance.
        train_file (string): Training file in .opf format.
        op (int): Operation to be chosen.
        value (float): Maximum value.

    """

    # Creates the training subgraph
    train = opf._readsubgraph(train_file.encode('utf-8'))

    # Performs the minimum cut
    opf._bestkmincut(train, 1, 100)

    # Eliminates according to chosen method
    eliminate(op, opf, train, value)

    # Performs the clustering
    opf._clustering(train)

    print('num of clusters %d' % train.contents.nlabels)

    # Writes the model file
    opf._writemodelfile(train, 'classifier.opf'.encode('utf-8'))

    # Writes the output file
    opf._writeoutputfile(train, 'training.dat.out'.encode('utf-8'))

    # Destroys the subgraph
    opf._destroysubgraph(train)


def _test(opf, test_file):
    """Performs the KNN classification.

    Args:
        opf (OPF): OPF class instance.
        test_file (string): Testing file in .opf format.

    """

    # Creates the testing subgraph
    test = opf._readsubgraph(test_file.encode('utf-8'))

    # Reads the training model file
    train = opf._readmodelfile('classifier.opf'.encode('utf-8'))

    # Performs the KNN classification
    opf._knn_classify(train, test)

    # Writes the output file
    opf._writeoutputfile(test, 'testing.dat.out'.encode('utf-8'))

    # Destroys the subgraph
    opf._destroysubgraph(test)


def _train(opf, train_file):
    """Performs the supervised OPF traning.

    Args:
        opf (OPF): OPF class instance.
        train_file (string): Training file in .opf format.

    """

    # Creates the training subgprah
    train = opf._readsubgraph(train_file.encode('utf-8'))

    # Performs the supervised OPF training
    opf._training(train)

    # Writes the model file
    opf._writemodelfile(train, 'classifier.opf'.encode('utf-8'))

    # Writes the output file
    opf._writeoutputfile(train, 'training.dat.out'.encode('utf-8'))

    # Destroys the subgraph
    opf._destroysubgraph(train)


def _classify(opf, test_file):
    """Performs the supervised OPF classification.

    Args:
        opf (OPF): OPF class instance.
        test_file (string): Testing file in .opf format.

    """

    # Creates the testing subgprah
    test = opf._readsubgraph(test_file.encode('utf-8'))

    # Reads the model file
    train = opf._readmodelfile('classifier.opf'.encode('utf-8'))

    # Performs the supervised OPF classification
    opf._classifying(train, test)

    # Writes the output file
    opf._writeoutputfile(test, 'testing.dat.out'.encode('utf-8'))

    # Destroys the subgraph
    opf._destroysubgraph(test)


def _acc(opf, test_file):
    """Performs the OPF accuracy computation.

    Args:
        opf (OPF): OPF class instance.
        test_file (string): Testing file in .opf format.

    """

    # Creates the testing subgraph
    test = opf._readsubgraph(test_file.encode('utf-8'))

    # Reads the output file
    opf._readoutputfile(test, 'testing.dat.out'.encode('utf-8'))

    # Performs the accuracy computation
    acc = opf._accuracy(test)

    print('Accuracy: %.2f' % (acc*100))

    # Destroys the subgraph
    opf._destroysubgraph(test)

    return acc
