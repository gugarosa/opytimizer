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
        print('Eliminating maxima in the graph with pdf below H (dome height)')

        # Creates the pointer to the function
        elimmaxbelowH = self._OPF.opf_ElimMaxBelowH

        # Gets the argument types
        elimmaxbelowH.argtypes = [POINTER(Subgraph), c_float]
        
        # Actually uses the function
        elimmaxbelowH(subgraph, h)
        

    def _elimmaxbelowA(self, subgraph, a):
        print('Eliminating maxima in the graph with pdf below A (area)')

        # Creates the pointer to the function
        elimmaxbelowA = self._OPF.opf_ElimMaxBelowArea

        # Gets the argument types
        elimmaxbelowA.argtypes = [POINTER(Subgraph), c_int]

        # Actually uses the function
        elimmaxbelowA(subgraph, a)

    def _elimmaxbelowV(self, subgraph, v):
        print('Eliminating maxima in the graph with pdf below V (volume)')

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


def dome_heigh(opf, subgraph, value):
    """

    Args:

    """

    Hmax = 0.0
    for i in range(subgraph.contents.nnodes):
        if (subgraph.contents.node[i].dens > Hmax):
            Hmax = subgraph.contents.node[i].dens
    opf._elimmaxbelowH(subgraph, (Hmax * value))


def dome_area(opf, subgraph, value):
    """

    Args:
    
    """

    opf._elimmaxbelowA(subgraph, int(value * subgraph.contents.nnodes))


def dome_volume(opf, subgraph, value):
    """

    Args:
    
    """

    Vmax = 0.0
    for i in range(subgraph.contents.nnodes):
        Vmax += subgraph.contents.node[i].dens
    opf._elimmaxbelowH(subgraph, int(
        value * (Vmax / subgraph.contents.nnodes)))


def eliminate_maxima(op, opf, subgraph, value):
    """

    Args:
    
    """

    switcher = {
        0: lambda: dome_heigh(opf, subgraph, value),
        1: lambda: dome_area(opf, subgraph, value),
        2: lambda: dome_volume(opf, subgraph, value),
    }
    return switcher.get(op, lambda: "ERROR: option invalid")()


def _cluster(opf, train_file, op, value):
    """

    Args:
    
    """

    train = opf._readsubgraph(train_file.encode('utf-8'))
    opf._bestkmincut(train, 1, 100)
    eliminate_maxima(op, opf, train, value)
    opf._clustering(train)
    print('num of clusters %d' % train.contents.nlabels)
    opf._writemodelfile(train, 'classifier.opf'.encode('utf-8'))
    opf._writeoutputfile(train, 'training.dat.out'.encode('utf-8'))
    opf._destroysubgraph(train)
    print('Train OK')


def _test(opf, test_file):
    """

    Args:
    
    """

    test = opf._readsubgraph(test_file.encode('utf-8'))
    train = opf._readmodelfile('classifier.opf'.encode('utf-8'))
    opf._knn_classify(train, test)
    opf._writeoutputfile(test, 'testing.dat.out'.encode('utf-8'))
    opf._destroysubgraph(test)
    print('Test OK')


def _train(opf, train_file):
    """

    Args:
    
    """

    train = opf._readsubgraph(train_file.encode('utf-8'))
    opf._training(train)
    opf._writemodelfile(train, 'classifier.opf'.encode('utf-8'))
    opf._writeoutputfile(train, 'training.dat.out'.encode('utf-8'))
    opf._destroysubgraph(train)
    print('Train OK')


def _classify(opf, test_file):
    """

    Args:
    
    """

    test = opf._readsubgraph(test_file.encode('utf-8'))
    train = opf._readmodelfile('classifier.opf'.encode('utf-8'))
    opf._classifying(train, test)
    opf._writeoutputfile(test, 'testing.dat.out'.encode('utf-8'))
    opf._destroysubgraph(test)
    print('Test OK')


def _acc(opf, test_file):
    """

    Args:
    
    """

    test = opf._readsubgraph(test_file.encode('utf-8'))
    opf._readoutputfile(test, 'testing.dat.out'.encode('utf-8'))
    acc = opf._accuracy(test)
    print('Acc: %.2f' % (acc*100))
    opf._destroysubgraph(test)
    return acc
