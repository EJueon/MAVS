# -*- coding: utf-8 -*- 
import sys
import pickle
import numpy as np
import tensorflow as tf
from keras import backend as K

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

from nrf_corpus import NRFCorpus, NRFElement
from nrf_fetcher import NRFFetcher
from nrf_coverage import NRFCoverage

from nrf_setting import ModelSetting



class CrashVisualization():
    def __init__(self, dataset, metrics):
        self.numExes = range(1)
        self.dirs = dict()
        self.modelSetting = ModelSetting('crashVisualization', dataset)
        self.epsilon = '0.2' #'0.1' if dataset == 'mnist' else '0.01'
        
        self.ncoveragedir = dataset + '/'
        self.attackdir = 'result/advattack'
        self.defensedir = '../fuzzing+defense-master/attack/test'
        
        self.metrics = metrics
        self.bitMetrics = {'kmnc', 'nbc', 'nc'}#['nc', 'kmnc', 'nbc', 'snac', 'bknc', 'tknc']
        self.attackMetrics = {'fgsm_%s' % self.epsilon, 'pgd_%s' % self.epsilon}
        
        # check : 개선 필요 : dir에 attackmetric 위치 명시
        for attackMetric in self.attackMetrics:
            self.dirs[attackMetric] = self.attackdir
        for bitMetric in self.bitMetrics:
            self.dirs[bitMetric] = self.ncoveragedir + bitMetric
        
        Fetcher = NRFFetcher(self.modelSetting.model, self.modelSetting.preprocess, 'tensorfuzz', 'penultimate', 100)
        self.fetcher = Fetcher.fetchFunction
        
        self.pca = PCA(n_components = 3)
        # Original dataset images
        self.coverages = self.get_penultimate(self.modelSetting.image, self.modelSetting.label)
        self.pca.fit(self.coverages)
        
        self.crashCounts = dict()
        self.crashImages = dict()
        self.uniqueCrashes = dict()
        self.numCrashes = dict()
        self.integerCrashes = dict()
        # check : 개선 필요
        for metric in self.metrics:
            self.crashCounts[metric] = list()
            self.crashImages[metric] = dict()
            self.uniqueCrashes[metric] = list()
            self.numCrashes[metric] = 0
            self.integerCrashes[metric] = 0
            for source in range(self.modelSetting.numLabels):
                self.crashCounts[metric].append(list())
                self.crashImages[metric][source] = list()
                for target in range(self.modelSetting.numLabels):
                    self.crashCounts[metric][source].append(0)
                    self.crashImages[metric][source].append(list())
     
    # check 코드 재확인 필요
    def get_penultimate(self, images, labels):
        batchmax = 100
        coverages = list()
        if len(images) > batchmax:
            for batchnum in range(int(len(images)/batchmax)):
                inputBatch = np.array(images[batchnum * batchmax:(batchnum + 1) * batchmax])
                inputLabels = np.array(labels[batchnum * batchmax:(batchnum + 1) * batchmax])
                coverage, metadata_, predict_batches, _ = self.fetcher(inputBatch, inputLabels)#self.fetcher.doFetch(inputBatch)
                coverages.extend(coverage[0])
            if len(images) % batchmax != 0:
                inputBatch = np.array(images[int(len(images) / batchmax) * batchmax:])
                inputLabels = np.array(labels[int(len(images) / batchmax) * batchmax:])
                coverage, metadata_, predict_batches, _ = self.fetcher(inputBatch, inputLabels)#self.fetcher.doFetch(inputBatch)
                coverages.extend(coverage[0])
        else:
                inputBatch = np.array(images)
                inputLabels = np.array(labels)
                coverage, metadata_, predict_batches, _ = self.fetcher(inputBatch, inputLabels)#self.fetcher.doFetch(inputBatch)
                coverages.extend(coverage[0])
        return coverages
    
    #neuron coverage check : attackmetric이랑 합칠 필요성 있음
    def measure_neuronMetrics(self):
        metrics = self.metrics & self.bitMetrics
        for metric in metrics:
            for num in self.numExes:
                with open('%s/%d/crash' %(self.dirs[metric], num), 'rb') as crashfile:
                    print(crashfile)
                    predictions = list()
                    readingCount = 0
                    while True:
                        try:
                            crash = pickle.load(crashfile)
                            sys.stdout.write('\rReading a crash element : %d' % readingCount)
                            sys.stdout.flush()
                            if crash.generation < 1: continue
                            self.crashImages[metric][crash.label][crash.prediction].append(crash.data)
                            readingCount += 1
                        except Exception as exception:
                            print('error ocurred: ', exception) 
                            break
    def measure_attackMetrics(self):
        metrics = self.metrics & self.attackMetrics
        for metric in metrics:
            with open('%s/%s_%s.pickle' % (self.dirs[metric], self.modelSetting.dataset, metric), 'rb') as crashfile:
                print(crashfile)
                crashes = pickle.load(crashfile)
                readingCount = 0
                predictions = list()
                for crash in crashes:
                    try:
                        coverage, metadata_, predict_batches, _ = self.fetcher(np.array([crash[0]]), np.array([crash[1]]))
                        if crash[1] != predict_batches[0]:# and crash[1] == 0:
                            self.crashImages[metric][crash[1]][predict_batches[0]].append(crash[0])
                            readingCount += 1
                    except Exception as exception:
                        print('error ocurred: ', exception) 
                        break
                
                
    # print_graph
    def print_results(self, metrics):
        self.metrics = metrics
        self.pcas = list()
        for source in range(self.modelSetting.numLabels):
            self.pcas.append(dict())
            for metric in self.metrics:
                self.pcas[source][metric] = list()
            for target in range(self.modelSetting.numLabels):
                for metric in self.metrics:
                    if len(self.crashImages[metric][source][target]) == 0: continue
                    print("metric: ", metric, "source: ", source, "target: ", target, len(self.crashImages[metric][source][target]))
                    self.crashCounts[metric][source][target] += 1
                    images = np.array(self.crashImages[metric][source][target])
                    labels = [source] * len(self.crashImages[metric][source][target])
                    self.coverages = self.get_penultimate(images, labels)
                        
                    if not len(self.pcas[source][metric]):
                        self.pcas[source][metric] = self.pca.transform(self.coverages)
                    else:
                        self.pcas[source][metric] = np.append(self.pcas[source][metric], self.pca.transform(self.coverages), axis = 0)
           
if __name__=="__main__":
    temp = CrashVisualization('mnist', {'nc'})
    temp.print_results()