from ctypes import *
import os
import sys
import numpy as np

class LibOPF:
    def __init__(self, OPF=None, dataset=None):
        self._OPF = CDLL(os.environ['OPF_DIR']+'/OPF.so')

    def wrap_function(lib, funcname, restype, argtypes):
        """Simplify wrapping ctypes functions"""
        func = lib.__getattr__(funcname)
        func.restype = restype
        func.argtypes = argtypes
        return func

class Set(Structure):
    pass

Set._fields_ = [
        ("elems", c_int),
        ("next", POINTER(Set))
    ]

class SNode(Structure):
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
    _fields_ = [
        ("node", POINTER(SNode)), \
        ("nnodes", c_int), \
        ("nfeats", c_int), \
        ("bestk", c_int), \
        ("nlabels", c_int), \
        ("df", c_float), \
        ("mindens", c_float), \
        ("maxdens", c_float), \
        ("k", c_float), \
        ("ordered_list_of_nodes", POINTER(c_int))
    ]

    def __init__(self, nnodes = None):
        super().__init__()

class OPF(LibOPF):
    def __init__(self, dataset=None):
        super().__init__()
        self._dataset = dataset

    def _readsubgraph(self, dataset):
        print('Reading data file ...')
        readsubgraph = self._OPF.ReadSubgraph
        readsubgraph.restype = POINTER(Subgraph)
        readsubgraph.argtypes = [c_char_p]
        g = readsubgraph(dataset)
        return g
        print('OK')

    def _writesubgraph(self, subgraph, file_name):
        print('Writing data file ...')
        writesubgraph = self._OPF.WriteSubgraph
        writesubgraph.argtypes = [POINTER(Subgraph), c_char_p]
        writesubgraph(subgraph, file_name)
        print('OK')

    def _destroysubgraph(self, subgraph):
        print('Deallocating memory ...')
        destroysubgraph = self._OPF.DestroySubgraph
        destroysubgraph.argtypes = [POINTER(POINTER(Subgraph))]
        destroysubgraph(subgraph)
        print('OK')

    def _writemodelfile(self, subgraph, file_name):
        print('Writing classifier\'s model file ...')
        writemodelfile = self._OPF.opf_WriteModelFile
        writemodelfile.argtypes = [POINTER(Subgraph), c_char_p]
        writemodelfile(subgraph, file_name)
        print('OK')

    def _readmodelfile(self, file_name):
        print('Reading classifier\'s model file ...')
        readmodelfile = self._OPF.opf_ReadModelFile
        readmodelfile.restype = POINTER(Subgraph)
        readmodelfile.argtypes = [c_char_p]
        g = readmodelfile(file_name)
        return g
        print('OK')

    def _modelfile2txt(self):
        print('Converting classifier.opf from binary to text file ...')
        modelfile2txt = self._OPF.opf_ModelFile2Txt
        modelfile2txt()
        print('OK')

    def _writeoutputfile(self, subgraph, file_name):
        print('Writing output file ...')
        writeoutputfile = self._OPF.opf_WriteOutputFile
        writeoutputfile.argtypes = [POINTER(Subgraph), c_char_p]
        writeoutputfile(subgraph, file_name)
        print('OK')

    def _readoutputfile(self, subgraph, file_name):
        print('Reading output file ...')
        readoutputfile = self._OPF.opf_ReadOutputFile
        readoutputfile.argtypes = [POINTER(Subgraph), c_char_p]
        readoutputfile(subgraph, file_name)
        print('OK')

    def _training(self, train):
        print('OPF supervised training ...')
        training = self._OPF.opf_OPFTraining
        training.argtypes = [POINTER(Subgraph)]
        training(train)
        print('OK')

    def _classifying(self, train, test):
        print('OPF classifying ...')
        classifying = self._OPF.opf_OPFClassifying
        classifying.argtypes = [POINTER(Subgraph)]
        classifying(test)
        print('OK')

    def _bestkmincut(self, train, k_min, k_max):
        print('Estimating the best k by minimum cut')
        bestkmincut = self._OPF.opf_BestkMinCut
        bestkmincut.argtypes = [POINTER(Subgraph), c_int, c_int]
        bestkmincut(train, k_min, k_max)
        print('OK')

    def _clustering(self, train):
        print('OPF clustering ...')
        clustering = self._OPF.opf_OPFClustering
        clustering.argtypes = [POINTER(Subgraph)]
        clustering(train)
        print('OK')

    def _knn_classify(self, train, test):
        print('Classifying test set ...')
        knn_classify = self._OPF.opf_OPFknnClassify
        knn_classify.argtypes = [POINTER(Subgraph), POINTER(Subgraph)]
        knn_classify(train, test)
        print('OK')

    def _elimmaxbelowH(self, subgraph, h):
        print('Eliminating maxima in the graph with pdf below H (dome height)')
        elimmaxbelowH = self._OPF.opf_ElimMaxBelowH
        elimmaxbelowH.argtypes = [POINTER(Subgraph), c_float]
        elimmaxbelowH(subgraph, h)
        print('OK')

    def _elimmaxbelowA(self, subgraph, a):
        print('Eliminating maxima in the graph with pdf below A (area)')
        elimmaxbelowA = self._OPF.opf_ElimMaxBelowArea
        elimmaxbelowA.argtypes = [POINTER(Subgraph), c_int]
        elimmaxbelowA(subgraph, a)
        print('OK')

    def _elimmaxbelowV(self, subgraph, v):
        print('Eliminating maxima in the graph with pdf below V (volume)')
        elimmaxbelowV = self._OPF.opf_ElimMaxBelowVolume
        elimmaxbelowV.argtypes = [POINTER(Subgraph), c_int]
        elimmaxbelowV(subgraph, v)
        print('OK')

    def _accuracy(self, subgraph):
        print('Computing accuracy ...')
        accuracy = self._OPF.opf_Accuracy
        accuracy.restype = c_float
        accuracy.argtypes = [POINTER(Subgraph)]
        result = accuracy(subgraph)
        return result

def dome_heigh(opf, subgraph, value):
    Hmax = 0.0
    for i in range(subgraph.contents.nnodes):
        if (subgraph.contents.node[i].dens > Hmax):
            Hmax = subgraph.contents.node[i].dens
    opf._elimmaxbelowH(subgraph, (Hmax * value))

def dome_area(opf, subgraph, value):
    opf._elimmaxbelowA(subgraph, int(value * subgraph.contents.nnodes))

def dome_volume(opf, subgraph, value):
    Vmax = 0.0
    for i in range(subgraph.contents.nnodes):
        Vmax += subgraph.contents.node[i].dens
    opf._elimmaxbelowH(subgraph, int(value * (Vmax / subgraph.contents.nnodes)))

def eliminate_maxima(op, opf, subgraph, value):
    switcher={
        0: lambda : dome_heigh(opf, subgraph, value),
        1: lambda : dome_area(opf, subgraph, value),
        2: lambda : dome_volume(opf, subgraph, value),
    }
    return switcher.get(op, lambda : "ERROR: option invalid")()

def _cluster(opf, train_file, op, value):
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
    test = opf._readsubgraph(test_file.encode('utf-8'))
    train = opf._readmodelfile('classifier.opf'.encode('utf-8'))
    opf._knn_classify(train, test)
    opf._writeoutputfile(test, 'testing.dat.out'.encode('utf-8'))
    opf._destroysubgraph(test)
    print('Test OK')

def _train(opf, train_file):
    train = opf._readsubgraph(train_file.encode('utf-8'))
    opf._training(train)
    opf._writemodelfile(train, 'classifier.opf'.encode('utf-8'))
    opf._writeoutputfile(train, 'training.dat.out'.encode('utf-8'))
    opf._destroysubgraph(train)
    print('Train OK')

def _classify(opf, test_file):
    test = opf._readsubgraph(test_file.encode('utf-8'))
    train = opf._readmodelfile('classifier.opf'.encode('utf-8'))
    opf._classifying(train, test)
    opf._writeoutputfile(test, 'testing.dat.out'.encode('utf-8'))
    opf._destroysubgraph(test)
    print('Test OK')

def _acc(opf, test_file):
    test = opf._readsubgraph(test_file.encode('utf-8'))
    opf._readoutputfile(test, 'testing.dat.out'.encode('utf-8'))
    acc = opf._accuracy(test)
    print('Acc: %.2f' % (acc*100))
    opf._destroysubgraph(test)
    return acc
