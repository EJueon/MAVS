# -*- coding: utf-8 -*- 
import sys
import pickle
import numpy as np
import tensorflow as tf
from keras import backend as K

from lib.queue import Seed
from utils.coverage import Coverage
from image_queue import ImageInputCorpus
from fuzzone import build_fetch_function

from utils.fetcher import Fetcher
from utils.setting import ModelSetting
from utils.coverage import Coverage

class CoverageRate():
    def __init__(self, dataset: str, metrics: str):
        self.elementtype = 'crash' #corpus, crash, parent
        self.numExes = range(1)
        self.ncoveragedir = dataset + '/'
        self.attackdir = 'result/advattack'
        self.defensedir = '../fuzzing+defense-master/attack/test'
        self.dirs = dict()
        self.modelSetting = ModelSetting('coverageRate', dataset)
        self.metrics = metrics
        self.bitMetrics = {'kmnc', 'nbc', 'nc'}
        self.attackMetrics = {'fgsm_0.2', 'pgd_0.2'}
        self.metrics_para = {'kmnc': 1000, 'bknc': 10, 'tknc': 10, 'nbc': 10, 'newnc': 10, 'nc': 0.75, 'snac': 10}

        for attackMetric in self.attackMetrics:
            self.dirs[attackMetric] = self.attackdir
        for bitMetric in self.bitMetrics:
            self.dirs[bitMetric] = self.ncoveragedir + bitMetric
            
        Fetcher = Fetcher(self.modelSetting.model, self.modelSetting.preprocess, 'tensorfuzz', 'penultimate', 100)
        self.fetcher = Fetcher.fetchFunction
            
        # coverage metrics에 필요한 프로파일링 정보 호출
        with open('%s/profile.pickle' % self.modelSetting.checkpoints[dataset], 'rb') as rfile:
            u = pickle._Unpickler(rfile)
            u.encoding = 'latin1'
            profile_dict = u.load()
            print("###profile read###")
            
        self.coverage_handler = dict() # coverage 
        self.dry_run_fetch = dict() # fetrch_function
        self.coverage_function = dict() # seed coverageQueue
        
        for metric in self.bitMetrics:
            
            self.coverage_handler[metric] = Coverage(model = self.modelSetting.model, criteria=metric, k = self.metrics_para[metric],
                                        profiling_dict=profile_dict,exclude_layer=['input', 'flatten', 'activation', 'batch', 'dropout'])
            self.dry_run_fetch[metric] = build_fetch_function(self.coverage_handler[metric], self.modelSetting.preprocess)
            self.coverage_function[metric] = self.coverage_handler[metric].update_coverage
        
        self.coverageQueue = dict() # The seed coverageQueue
        for metric in self.metrics:
            self.coverageQueue[metric] = dict()
            for metric2 in self.bitMetrics:
                self.coverageQueue[metric][metric2] = ImageInputCorpus('./tempdir', 0, 'prob', self.coverage_handler[metric2].total_size, metric2)
         
    # neuron coverage : check attack metric과 합쳐야함
    def measure_neuronMetrics(self):
        metrics = self.metrics & self.bitMetrics
        numLabels = 1 if self.elementtype == 'parent' else self.modelSetting.numLabels 
        
        for metric in metrics:
            images = list()
            # 1. reading corpus
            readingCount = 0 #read corpus number
            for num in self.numExes:
                with open('%s/%d/%s' % (self.dirs[metric], num, self.elementtype), 'rb') as corpusfile:
                    print(corpusfile.name)
                    while True:
                        try:
                            corpus = pickle.load(corpusfile)
                            sys.stdout.write('\r Reading elements : number %d'%(readingCount))
                            sys.stdout.flush()
                            readingCount += 1
                            if type(corpus) == type(list()): data = corpus[0]
                            else: data = corpus.data
                            images.append(data)
                        except Exception as exception:
                            print('error ocurred: ', exception) 
                            break
                print()
            readingCount = 0
            # 2. measuring coverage
            for image in images:
                sys.stdout.write('\rMeasuring coverage of elements : number %d'%(readingCount))
                sys.stdout.flush()
                readingCount += 1
                input_batches = [image]
                for metric2 in self.bitMetrics:
                    # Predict the mutant and obtain the outputs
                    # coverage_batches is the output of internal layers and metadata_batches is the output of the prediction result
                    coverage_batches,  metadata_batches = self.dry_run_fetch[metric2]((0,input_batches,0,0,0,0))
                    # Based on the output, compute the coverage information
                    coverage_list = self.coverage_function[metric2](coverage_batches)
                    # Create a new seed
                    input = Seed(0, coverage_list[0], None, None, None, None)
                    self.coverageQueue[metric][metric2].has_new_bits(input)
            print()
            # 3. Saving coverage
            for metric2 in self.bitMetrics:
                with open('result/coverage/%s.%s%d-%d.%s' % (metric, self.elementtype, min(list(self.numExes)), max(list(self.numExes)), metric2), 'wb') as coveragefile:
                    print(coveragefile)
                    try:
                        print(type(self.coverageQueue[metric][metric2]))
                        corpus = pickle.dump([self.coverageQueue[metric][metric2].virgin_bits, self.coverageQueue[metric][metric2].total_cov], coveragefile)
                    except Exception as e:
                        print(e)
            print()
    
    def measure_attackMetrics(self):
        metrics = self.metrics & self.attackMetrics
        print(metrics)
        for metric in metrics:
            images = list()
            with open('%s/%s_%s.pickle' % (self.dirs[metric], self.modelSetting.dataset, metric), 'rb') as crashfile:
                crashes = pickle.load(crashfile)
                readingCount = 0
                for crash in crashes:
                    coverage, metadata_, predict_batches, _ = self.fetcher(np.array([crash[0]]), np.array([crash[1]]))
                    if crash[1] != predict_batches[0]:
                        images.append(crash[0])
                    readingCount += 1
            readingCount = 0
            for image in images:
                sys.stdout.write('\rMeasuring coverage of elements : number %d'%(readingCount))
                sys.stdout.flush()
                readingCount += 1
                input_batches = [image]
                for metric2 in self.bitMetrics:
                    # Predict the mutant and obtain the outputs
                    # coverage_batches is the output of internal layers and metadata_batches is the output of the prediction result
                    coverage_batches,  metadata_batches = self.dry_run_fetch[metric2]((0,input_batches,0,0,0,0))
                    # Based on the output, compute the coverage information
                    coverage_list = self.coverage_function[metric2](coverage_batches)
                    # Create a new seed
                    input = Seed(0, coverage_list[0], None, None, None, None)
                    self.coverageQueue[metric][metric2].has_new_bits(input)
            print()
            # 3. Saving coverage
            for metric2 in self.bitMetrics:
                with open('result/coverage/%s.%s' % (metric, metric2), 'wb') as coveragefile:
                    print(coveragefile)
                    try:
                        print(type(self.coverageQueue[metric][metric2]))
                        corpus = pickle.dump([self.coverageQueue[metric][metric2].virgin_bits, self.coverageQueue[metric][metric2].total_cov], coveragefile)
                    except Exception as e:
                        print(e)
            print()
            
                    
    # Print Whole Coverage
    def print_metrics(self):
        for metric in self.metrics:
            for metric2 in self.bitMetrics:
                self.coverageQueue[metric][metric2].log()
                coverageResult = round(float(self.coverageQueue[metric][metric2].total_cov - np.count_nonzero(self.coverageQueue[metric][metric2].virgin_bits == 0xFF)) * 100 / self.coverageQueue[metric][metric2].total_cov, 2)
                print('%s found %s coverage of %s' % (metric, metric2, coverageResult))
                        
                    
                
# TEST
# if __name__=="__main__":
#     temp = CoverageRate('mnist', 'nc') 
        
        
            
            
        
        
    

  
 
