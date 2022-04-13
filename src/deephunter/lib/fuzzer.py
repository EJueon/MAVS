import tensorflow as tf
import gc
import sys
import time
import copy
import pickle
import numpy as np
from datetime import datetime
class Fuzzer(object):
    """Class representing the fuzzer itself."""

    def __init__(
        self,
        corpus,
        coverage_function,
        metadata_function,
        objective_function,
        mutation_function,
        fetch_function,
        iterate_function,
        plot = True,
        model = None,
        exec = None
    ):
        """Init the class.

    Args:
      corpus: An InputCorpus object.
      coverage_function: a function that does CorpusElement -> Coverage.
      metadata_function: a function that does CorpusElement -> Metadata.
      objective_function: a function that checks if a CorpusElement satisifies
        the fuzzing objective (e.g. find a NaN, find a misclassification, etc).
      mutation_function: a function that does CorpusElement -> Metadata.
      fetch_function: grabs numpy arrays from the TF runtime using the relevant
        tensors.
    Returns:
      Initialized object.
    """
        self.plot = plot
        self.queue = corpus
        self.coverage_function = coverage_function
        self.metadata_function = metadata_function
        self.objective_function = objective_function
        self.mutation_function = mutation_function
        self.fetch_function = fetch_function
        self.iterate_function = iterate_function

        class recoder():
            def __init__(self):
                # The number of each data
                self.noIteration = list()
                self.periodTime = list()
                self.noCrash = list()
                self.noElement = list()
                self.noCrashParents = list()
                self.crashParentsCount = 0
                self.noErrorCategories = list()
                
        self.timeRecoder = recoder()
        self.iterRecoder = recoder()
        self.recoders = [self.timeRecoder, self.iterRecoder]

        self.crashArray = np.array([])
        self.crashDict = dict()
        self.crashes = 0
        self.crashParents = list()
        self.errorCategories = dict()

        self.parents = list()

        if exec >= 0:
            self.resultDir = '../result/%s/%d' % (model, exec)
        else:
            self.resultDir = '../result/%s' % model
        self.crashFile = open('%s/crash' % self.resultDir, 'wb')
        #self.parentFile = open('%s/parent' % self.resultDir, 'wb')


    def recordCurrentState_time(self):
        self.timeRecoder.noIteration.append(self.iteration)
        self.timeRecoder.noCrash.append(self.crashes)
        self.timeRecoder.noElement.append(self.queue.total_queue)
        self.timeRecoder.noCrashParents.append(self.timeRecoder.crashParentsCount)
        self.timeRecoder.crashParentsCount = 0
        self.timeRecoder.noErrorCategories.append(np.sum([len(self.queue.errorCategories[sample]) for sample in self.queue.errorCategories.keys()]))


    def recordCurrentState_iteration(self, time):
        self.iterRecoder.noIteration.append(self.iteration)
        self.iterRecoder.periodTime.append(time)
        self.iterRecoder.noCrash.append(self.crashes)
        self.iterRecoder.noElement.append(self.queue.total_queue)
        self.iterRecoder.noCrashParents.append(self.iterRecoder.crashParentsCount)
        self.iterRecoder.crashParentsCount = 0
        self.iterRecoder.noErrorCategories.append(np.sum([len(self.queue.errorCategories[sample]) for sample in self.queue.errorCategories.keys()]))

    def saveResult(self, time):
        # Save iteration information
        import csv
        import pickle
        with open('%s/timeRecode.csv' % self.resultDir, 'w', newline='') as wfile:
            writer = csv.writer(wfile)
            
            writer.writerow(['No. Iteration'] + self.timeRecoder.noIteration)
            writer.writerow(['No. Crash'] + self.timeRecoder.noCrash)
            writer.writerow(['No. Elements'] + self.timeRecoder.noElement)
            writer.writerow(['No. Unique CrashParents'] + self.timeRecoder.noCrashParents)
            writer.writerow(['No. Error Category'] + self.timeRecoder.noErrorCategories)
            writer.writerow(['Time'] + [time])

        with open('%s/iterationRecode.csv' % self.resultDir, 'w', newline='') as wfile:
            writer = csv.writer(wfile)
            
            writer.writerow(['No. Iteration'] + self.iterRecoder.noIteration)
            writer.writerow(['No. Period Time'] + self.iterRecoder.periodTime)
            writer.writerow(['No. Crash'] + self.iterRecoder.noCrash)
            writer.writerow(['No. Elements'] + self.iterRecoder.noElement)
            writer.writerow(['No. Unique Crash Parents'] + self.iterRecoder.noCrashParents)
            writer.writerow(['No. Error Category'] + self.iterRecoder.noErrorCategories)
            writer.writerow(['Time'] + [time])

        # Save corpus information
        with open('%s/corpus' % self.resultDir, 'wb') as wfile:
            for element in self.queue.queue:
                pickle.dump(element, wfile)
        # Save crash information
        #with open('D:/Development/experiment/deephunter/result/mnist/crash', 'wb') as wfile:
        #    for crash in self.crashes:
        #        pickle.dump(crash, wfile)


    def plotCrash(self, element):
    
        from matplotlib import pyplot as plt

        def plotimg(img):
            convImg = (np.reshape(img, (28, 28)))#.astype(np.uint8)
            plt.figure(figsize=(3, 3))
            plt.imshow(convImg, interpolation='nearest', cmap='gray')
            plt.show()

        plotimg(element.data)


    def loop(self, iterations, timeout=None):
        """Fuzzes a machine learning model in a loop, making *iterations* steps."""
        startTime = datetime.now()
        if iterations and timeout:
            import signal

            class Scheduler(object):
                def __init__(self, fuzzer):
                    self._tasks = [(300, self._heartbeat), (timeout, self._stop)]
                    self._tick = 0
                    self.fuzzer = fuzzer
                    self.stopped = False

                def _heartbeat(self):
                    self.fuzzer.recordCurrentState_time()

                def _stop(self):
                    print('\nFuzzing reached timeout, Total iteration = %d' % self.fuzzer.iteration)
                    self.fuzzer.stopped = True
                    self.stopped = True

                def _execute(self, signum, stack):
                    if self.stopped:
                        return

                    self._tick += 300
                    for period, task in self._tasks:
                        if 0 == self._tick % period:
                            task()
                    signal.alarm(300)

                def start(self):
                    signal.signal(signal.SIGALRM, self._execute)
                    signal.alarm(300)

            s = Scheduler(self)
            s.start()

            self.iteration = 0
            self.stopped = False

            while True:
                if self.iteration >= iterations and self.stopped == True:
                    break

                #if len(self.queue.queue) < 1 or self.iteration >= iterations:
                #    break
                if self.iteration % 100 == 0:
                    tf.logging.info("fuzzing iteration: %s", self.iteration)
                    gc.collect()


                parent = self.queue.select_next()
                # Get a mutated batch for each input
                mutated_data_batches = self.mutation_function(parent)
                # Grab the coverage and metadata for mutated batch
                coverage_batches, metadata_batches = self.fetch_function(
                    mutated_data_batches
                )

                # Plot the data
                if self.plot:
                    self.queue.plot_log(self.iteration)


                if coverage_batches is not None and len(coverage_batches) > 0:
                    # Get the coverage - one from each batch element
                    mutated_coverage_list = self.coverage_function(coverage_batches)

                    # Get the metadata objects - one from each batch element
                    mutated_metadata_list = self.metadata_function(metadata_batches)

                    # Check for each mutant and decide whether it will be saved
                    bug_found, cov_inc, crashes = self.iterate_function(self.queue, parent.root_seed, parent, mutated_coverage_list, mutated_data_batches,mutated_metadata_list, self.objective_function)
                    del mutated_coverage_list
                    del mutated_metadata_list
                else:
                    bug_found = False
                    cov_inc = False

                if bug_found:
                    for crash in crashes:

                        update = False
                        expandCrash = crash.data.flatten()
                        expandCrash = np.expand_dims(expandCrash, axis=0)
                        if not len(self.crashArray):
                            self.crashArray = expandCrash
                            update = True
                        else:
                            dupCrash = self.crashArray - crash.data.flatten()
                            result = np.count_nonzero(np.count_nonzero(dupCrash, axis=1))
                            if result == len(self.crashArray):
                                self.crashArray = np.append(self.crashArray, expandCrash, axis=0)
                                update = True

                        if update:
                            #print(crash.prediction)
                            #self.plotCrash(crash)
                            if crash.prediction not in self.queue.errorCategories[crash.sampleNo].keys():
                                self.queue.errorCategories[crash.sampleNo][crash.prediction] = 0
                            self.queue.errorCategories[crash.sampleNo][crash.prediction] += 1

                            #if self.queue.errorCategories[crash.sampleNo][crash.prediction] > 100:
                            #    continue
                            if crash.parent not in self.parents:
                                self.parents.append(crash.parent)
                                for recoder in self.recoders:
                                    recoder.crashParentsCount += 1
                            #    crashParent = copy.deepcopy(crash.parent)
                            #    crashParent.parent = None
                            #    pickle.dump(crashParent, self.parentFile)
                            #    del crashParent
                            #crash.parent = crash.oldest_ancestor()[0]#crash = [crash.data, crash.label, crash.prediction, crash.sampleNo]#crash.parent = None
                            pickle.dump(crash, self.crashFile)#pickle.dump([crash.data, crash.label, crash.oldest_ancestor()[0].data], self.crashFile)#
                            self.crashes += 1

                ''' backup
                if bug_found:
                    if crashes[0].parent not in self.crashParents:
                        self.crashParents.append(crashes[0].parent)
                        for recoder in self.recoders:
                            recoder.crashParentsCount += 1
                    for crash in crashes:
                        keyTuple = (np.mean(crash.data), np.std(crash.data), crash.label)
                        if keyTuple not in self.crashDict.keys():
                            self.crashDict[keyTuple] = crash.data
                            if crash.prediction not in self.queue.errorCategories[crash.sampleNo]:
                                self.queue.errorCategories[crash.sampleNo].append(crash.prediction)
                            if crash.parent not in self.parents:
                                self.parents.append(crash.parent)
                                crashParent = copy.deepcopy(crash.parent)
                                crashParent.parent = None
                                pickle.dump(crashParent, self.parentFile)
                                del crashParent
                            crash.parent = crash.oldest_ancestor()[0]#crash = [crash.data, crash.label, crash.prediction, crash.sampleNo]#crash.parent = None
                            pickle.dump(crash, self.crashFile)
                            self.crashes += 1'''

                self.queue.fuzzer_handler(self.iteration, parent, bug_found, cov_inc)
                self.iteration += 1
                if self.iteration % 5000 == 0:
                    endTime = datetime.now()
                    self.recordCurrentState_iteration(endTime - startTime)
                if self.iteration == iterations:
                    endTime = datetime.now()
                    print('\nFuzzing reached max iteration, execution time = %s' % (endTime - startTime))

                del mutated_data_batches
                del coverage_batches
                del metadata_batches
            return None


        elif iterations and not timeout:
            self.iteration = 0
            while True:

                if len(self.queue.queue) < 1 or self.iteration >= iterations:
                    break
                if self.iteration % 100 == 0:
                    tf.logging.info("fuzzing iteration: %s", self.iteration)
                    gc.collect()


                parent = self.queue.select_next()
                # Get a mutated batch for each input
                mutated_data_batches = self.mutation_function(parent)
                # Grab the coverage and metadata for mutated batch
                coverage_batches, metadata_batches = self.fetch_function(
                    mutated_data_batches
                )

                # Plot the data
                if self.plot:
                    self.queue.plot_log(self.iteration)


                if coverage_batches is not None and len(coverage_batches) > 0:
                    # Get the coverage - one from each batch element
                    mutated_coverage_list = self.coverage_function(coverage_batches)

                    # Get the metadata objects - one from each batch element
                    mutated_metadata_list = self.metadata_function(metadata_batches)

                    # Check for each mutant and decide whether it will be saved
                    bug_found, cov_inc, crashes = self.iterate_function(self.queue, parent.root_seed, parent, mutated_coverage_list, mutated_data_batches,mutated_metadata_list, self.objective_function)
                    del mutated_coverage_list
                    del mutated_metadata_list
                else:
                    bug_found = False
                    cov_inc = False

                if bug_found:
                    for crash in crashes:
                        #keyTuple = (np.mean(crash.data), np.std(crash.data), crash.label)

                        update = False
                        expandCrash = crash.data.flatten()
                        expandCrash = np.expand_dims(expandCrash, axis=0)
                        if not len(self.crashArray):
                            self.crashArray = expandCrash
                            update = True
                        else:
                            dupCrash = self.crashArray - crash.data.flatten()
                            result = np.count_nonzero(np.count_nonzero(dupCrash, axis=1))
                            if result == len(self.crashArray):
                                self.crashArray = np.append(self.crashArray, expandCrash, axis=0)
                                update = True

                        if update:#keyTuple not in self.crashDict.keys():
                            #self.crashDict[keyTuple] = crash.data
                            if crash.prediction not in self.queue.errorCategories[crash.sampleNo]:
                                self.queue.errorCategories[crash.sampleNo].append(crash.prediction)
                            if crash.parent not in self.parents:
                                self.parents.append(crash.parent)
                                for recoder in self.recoders:
                                    recoder.crashParentsCount += 1
                            #    crashParent = copy.deepcopy(crash.parent)
                            #    crashParent.parent = None
                            #    pickle.dump(crashParent, self.parentFile)
                            #    del crashParent
                            #crash.parent = crash.oldest_ancestor()[0]#crash = [crash.data, crash.label, crash.prediction, crash.sampleNo]#crash.parent = None
                            pickle.dump(crash, self.crashFile)
                            self.crashes += 1

                self.queue.fuzzer_handler(self.iteration, parent, bug_found, cov_inc)
                self.iteration += 1

                if self.iteration % 5000 == 0:
                    endTime = datetime.now()
                    self.recordCurrentState_iteration(endTime - startTime)
                if self.iteration == iterations:
                    endTime = datetime.now()
                    print('\nFuzzing reached max iteration, execution time = %s' % (endTime - startTime))

                del mutated_data_batches
                del coverage_batches
                del metadata_batches
            return None

        else:
            exit()
