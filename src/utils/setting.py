import pickle
import numpy as np
from keras.models import load_model


class ModelSetting():
    def __init__(self, analysis: str, dataset: str='mnist'):
        self.analysis = analysis
        self.checkpoints = { 'mnist': 'ckpt/lenet5', 'cifar10': 'ckpt/vgg19', 'gtsrb': 'ckpt/micronnet'}
        self.targetModelSet = { 'mnist': 'lenet5', 'cifar10': 'vgg19', 'gtsrb': 'micronnet'}
        self.datasetdir = {'mnist': 'data/mnist/test_1000.data', 'cifar10': 'data/cifar10/test_1000.data','gtsrb': 'data/gtsrb/test_5000.data'}
        self.dataset = dataset
        self.targetModel = self.targetModelSet[dataset]
        self.numLabels = 0
        self.filterVector = None
        self.image, self.label = self.loadDataset()
        self.preprocess = self.preprocessing
        self.model = load_model('%s/model.h5' % self.checkpoints[self.dataset])
        print('###Load checkpoint complete.###')
    
    def preprocessing(self, data):
        if self.analysis == 'coverageRate': 
            data = np.copy(data)
            if self.dataset == 'mnist':
                data = np.reshape((data/255) - 0.5, (-1, 28, 28, 1))
            elif self.dataset == 'cifar10' or dataset == 'GTSRB':
                data = np.interp(data, (0,255), (-0.5, +0.5))
        elif self.analysis == 'crashVisualization':
            if self.dataset == 'mnist':
                data = np.reshape((data / 255) - 0.5, (-1, 28, 28, 1))
            elif self.dataset == 'cifar10':
                data = data
                data /= 255 
                data -= 0.5
            elif self.dataset == 'GTSRB':
                data = np.interp(data, (0, 255), (-0.5, +0.5))
        return data   
    
    def loadDataset(self):
        if self.dataset == 'mnist':
            self.numLabels = 10 
            filterVector = np.ones((28, 28))
            self.filterVector = np.reshape(filterVector, (28, 28, 1))    
        elif self.dataset == 'cifar10':
            self.numLabels = 10
            self.filterVector = np.ones((32, 32, 3))
        elif self.dataset == 'gtsrb':
            self.numlabels = 43
            self.filterVector = np.ones((48, 48, 3))
        with open(self.datasetdir[self.dataset], 'rb') as rfile:
            testData = pickle.load(rfile)
            print('###Brought '+ self.dataset + ' dataset.###') 
        return testData[0], testData[1]
    
 
    
    