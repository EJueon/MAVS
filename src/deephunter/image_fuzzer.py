from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

import argparse, pickle
import shutil

from keras.models import load_model
import tensorflow as tf
import os

sys.path.append('../')

from keras import Input
from deephunter.coverage import Coverage

from keras.applications import MobileNet, VGG19, ResNet50
from keras.applications.vgg16 import preprocess_input

import random
import time
import numpy as np
from deephunter.image_queue import ImageInputCorpus
from deephunter.fuzzone import build_fetch_function

from lib.queue import Seed
from lib.fuzzer import Fuzzer

from deephunter.mutators import Mutators
from keras.utils.generic_utils import CustomObjectScope

def imagenet_preprocessing(input_img_data):
    temp = np.copy(input_img_data)
    temp = np.float32(temp)
    qq = preprocess_input(temp)
    return qq

def mnist_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = np.reshape((np.array(temp) / 255) - 0.5, (-1, 28, 28, 1))
    return temp

def cifar_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = np.interp(temp, (0, 255), (-0.5, +0.5))
    return temp

def gtsrb_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = np.interp(temp, (0, 255), (-0.5, +0.5))
    return temp

model_weight_path = {
    'vgg16': "./profile/cifar10/models/vgg16.h5",
    'vgg19': "./profile/cifar10/models/vgg19.h5",
    'resnet20': "./profile/cifar10/models/resnet.h5",
    'lenet1': "./profile/mnist/models/lenet1.h5",
    'lenet4': "./profile/mnist/models/lenet4.h5",
    'lenet5': "./profile/mnist/models/lenet5.h5",
    'micronnet': "./profile/gtsrb/models/micronnet.h5"
}

model_profile_path = {
    'vgg16': "./profile/cifar10/profiling/vgg16/0_50000.pickle",
    'vgg19': "./profile/cifar10/profiling/vgg19/0_50000.pickle",
    'resnet20': "./profile/cifar10/profiling/resnet20/0_50000.pickle",
    'lenet1': "./profile/mnist/profiling/lenet1/0_60000.pickle",
    'lenet4': "./profile/mnist/profiling/lenet4/0_60000.pickle",
    'lenet5': "./profile/mnist/profiling/lenet5/0_60000.pickle",
    'mobilenet': "./profile/imagenet/profiling/mobilenet_merged.pickle",
    'resnet50': "./profile/imagenet/profiling/resnet50_merged.pickle",
    'micronnet': "./profile/gtsrb/profiling/profile.pickle"
}

preprocess_dic = {
    'vgg16': cifar_preprocessing,
    'vgg19': cifar_preprocessing,
    'resnet20': cifar_preprocessing,
    'lenet1': mnist_preprocessing,
    'lenet4': mnist_preprocessing,
    'lenet5': mnist_preprocessing,
    'mobilenet': imagenet_preprocessing,
    'resnet50': imagenet_preprocessing,
    'micronnet': gtsrb_preprocessing
}

shape_dic = {
    'vgg16': (32,32,3),
    'vgg19': (32,32,3),
    'resnet20': (32,32,3),
    'lenet1': (28,28,1),
    'lenet4': (28,28,1),
    'lenet5': (28,28,1),
    'mobilenet': (224, 224,  3),
    'resnet50': (224, 224,  3),
    'micronnet': (48, 48,  3)
}
metrics_para = {
    'kmnc': 1000,
    'bknc': 10,
    'tknc': 10,
    'nbc': 10,
    'newnc': 10,
    'nc': 0.75,
    'snac': 10
}
execlude_layer_dic = {
    'vgg16': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'vgg19': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'resnet20': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'lenet1': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'lenet4': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'lenet5': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'mobilenet': ['input', 'flatten', 'padding', 'activation', 'batch', 'dropout',
                  'bn', 'reshape', 'relu', 'pool', 'concat', 'softmax', 'fc'],
    'resnet50': ['input', 'flatten', 'padding', 'activation', 'batch', 'dropout', 'bn',
                              'reshape', 'relu', 'pool', 'concat', 'add', 'res4', 'res5'],
    'micronnet': ['input', 'flatten', 'activation', 'batch', 'dropout']
}
mutations = {
    'tensorfuzz': Mutators.tensorfuzz_mutation,
    'deephunter': Mutators.image_random_mutate
}
def metadata_function(meta_batches):
    return meta_batches

def image_mutation_function(batch_num, mutation):
    # Given a seed, randomly generate a batch of mutants
    def func(seed):
        return mutations[mutation](seed, batch_num)
    return func



def objective_function(seed, names):
    metadata = seed.metadata

    ground_truth = seed.ground_truth
    assert(names is not None)
    results = []
    if len(metadata)  == 1:
        # To check whether it is an adversarial sample
        if metadata[0] != ground_truth:
           results.append('')
    else:
        # To check whether it has different results between original model and quantized model
        # metadata[0] is the result of original model while metadata[1:] is the results of other models.
        # We use the string adv to represent different types;
        # adv = '' means the seed is not an adversarial sample in original model but has a different result in the
        # quantized version.  adv = 'a' means the seed is adversarial sample and has a different results in quan models.
        if metadata[0] == ground_truth:
            adv = ''
        else:
            adv = 'a'
        count = 1
        while count < len(metadata):
            if metadata[count] != metadata[0]:
                results.append(names[count]+adv)
            count += 1

    # results records the suffix for the name of the failed tests
    return results

def iterate_function(names):
    def func(queue, root_seed, parent, mutated_coverage_list, mutated_data_batches, mutated_metadata_list,
                         objective_function):

        ref_batches, batches, cl_batches, l0_batches, linf_batches, tids = mutated_data_batches

        successed = False
        bug_found = False
        crashes = list()
        # For each mutant in the batch, we will check the coverage and whether it is a failed test
        for idx in range(len(mutated_coverage_list)):

            input = Seed(cl_batches[idx], mutated_coverage_list[idx], root_seed, parent, mutated_metadata_list[:, idx],
                         parent.ground_truth, parent.sampleNo, parent.generation+1, l0_batches[idx], linf_batches[idx], data=batches[idx], tid=parent.tid+[tids[idx]])

            # The implementation for the isFailedTest() in Algorithm 1 of the paper
            results = objective_function(input, names)

            if len(results) > 0:
                # We have find the failed test and save it in the crash dir.
                for i in results:
                    queue.save_if_interesting(input, batches[idx], True, suffix=i)
                bug_found = True
                crashes.append(input)
            else:

                new_img = np.append(ref_batches[idx:idx + 1], batches[idx:idx + 1], axis=0)
                # If it is not a failed test, we will check whether it has a coverage gain
                result = queue.save_if_interesting(input, new_img, False)
                successed = successed or result
        return bug_found, successed, crashes

    return func


def dry_run(model, indir, fetch_function,coverage_function, queue, clss):

    #seed_lis = os.listdir(indir)
    ## Read each initial seed and analyze the coverage
    #for seed_name in seed_lis:
    #    tf.logging.info("Attempting dry run with '%s'...", seed_name)
    #    path = os.path.join(indir, seed_name)
    #    img = np.load(path)
    #    # Each seed will contain two images, i.e., the reference image and mutant (see the paper)
    #    input_batches = img[1:2]

        

        
    # 데이터셋 가져오기
    if model in ['lenet1', 'lenet4', 'lenet5']:
        #from tensorflow.examples.tutorials.mnist import input_data

        #mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

        #test = mnist.test.next_batch(10000)

        #testImage = np.reshape(test[0] - 0.5, (-1, 28, 28, 1))
        #testLabel = test[1]
        

        with open('data/mnist/test_1000.data', 'rb') as rfile:
            test = pickle.load(rfile)
        testImage = test[0]
        testLabel = test[1]
    elif model in ['vgg16', 'vgg19']:
        #from keras import datasets
        #from keras import backend
        #(trainImage, trainLabel), (testImage, testLabel) = datasets.cifar10.load_data()
        #img_rows, img_cols, img_ch = trainImage.shape[1:]

        #if backend.image_data_format() == 'channels_first':
        #    trainImage = trainImage.reshape(trainImage.shape[0], 1, img_rows, img_cols)
        #    testImage = testImage.reshape(testImage.shape[0], 1, img_rows, img_cols)
        #    input_shape = (1, img_rows, img_cols)
        #else:
        #    trainImage = trainImage.reshape(trainImage.shape[0], img_rows, img_cols, img_ch)
        #    testImage = testImage.reshape(testImage.shape[0], img_rows, img_cols, img_ch)
        #    input_shape = (img_rows, img_cols, 1)

        #trainImage = trainImage.astype('float32')
        #testImage = testImage.astype('float32')
        #trainImage /= 255
        #trainImage -= 0.5
        #testImage /= 255
        #testImage -= 0.5

        #testLabel = np.reshape(testLabel, (10000,))
        
        with open('data/cifar10/test_1000.data', 'rb') as rfile:
            test = pickle.load(rfile)
        testImage = test[0]
        testLabel = test[1]
    elif model in ['micronnet']:
        #import gtsrb_pre_data
        #from keras.preprocessing.image import ImageDataGenerator

        #x_train, y_train,x_val, y_val, x_test, y_test = gtsrb_pre_data.pre_data()


        #img_rows, img_cols, img_ch = x_train.shape[1:]

        ##y_train = keras.utils.to_categorical(y_train, numLabels)
        ##y_val = keras.utils.to_categorical(y_val, numLabels)
        ## y_test = keras.utils.to_categorical(y_test, numLabels)

        #train_datagen = ImageDataGenerator(
        #        rotation_range=40,
        #        horizontal_flip=False,
        #        width_shift_range=0.2,
        #        height_shift_range=0.2,
        ##        brightness_range=[0.8,1.0],
        #        shear_range=0.2,
        #        zoom_range=0.2,
        #        fill_mode='nearest',
        #        )
        #validation_datagen = ImageDataGenerator()
        #test_datagen = ImageDataGenerator()

        #aug_data=train_datagen.flow(x_train,y_train,batch_size=50)
        #val_data=validation_datagen.flow(x_val,y_val,batch_size=50)

        #testImage = np.array(x_test, dtype=np.float32)
        #testLabel = y_test
        ## print(y_test.shape)

        #Activate when loading dataset
        with open('data/gtsrb/test_5000.data', 'rb') as rfile:
            test = pickle.load(rfile)
        testImage = np.array(test[0], dtype=np.float32)
        testLabel = test[1]
    else:
        pass

    for index in range(len(testImage)):
        input_batches = testImage[index:index+1]
        label_batches = testLabel[index:index+1]

        # Predict the mutant and obtain the outputs
        # coverage_batches is the output of internal layers and metadata_batches is the output of the prediction result
        coverage_batches,  metadata_batches = fetch_function((0,input_batches,0,0,0,0))
        # Based on the output, compute the coverage information
        coverage_list = coverage_function(coverage_batches)
        metadata_list = metadata_function(metadata_batches)
        if metadata_list[0] == label_batches:
            # Create a new seed
            seed_name = 'input%d'%index
            input = Seed(clss, coverage_list[0], seed_name, None, metadata_list[0][0],metadata_list[0][0], data=input_batches[0], sampleNo=index, generation=0, tid=[-1])
            new_img = np.append(input_batches, input_batches, axis=0)
            # Put the seed in the queue and save the npy file in the queue dir
            queue.save_if_interesting(input, new_img, False, True, seed_name)
            queue.errorCategories[index] = dict()#queue.errorCategories[index] = list()




if __name__ == '__main__':

    start_time = time.time()

    tf.logging.set_verbosity(tf.logging.INFO)
    random.seed(time.time())

    parser = argparse.ArgumentParser(description='coverage guided fuzzing for DNN')

    parser.add_argument('-i', help='input seed directory')
    parser.add_argument('-o', help='output directory')

    parser.add_argument('-model', help="target model to fuzz", choices=['vgg16', 'resnet20', 'mobilenet', 'vgg19',
                                                                     'resnet50','lenet1', 'lenet4', 'lenet5', 'micronnet'])
    parser.add_argument('-criteria', help="set the criteria to guide the fuzzing",
                        choices=['nc', 'kmnc', 'nbc', 'snac', 'bknc', 'tknc'], default='kmnc')
    parser.add_argument('-batch_num', help="the number of mutants generated for each seed", type=int, default=20)
    parser.add_argument('-max_iteration', help="maximum number of fuzz iterations", type=int, default=10000000)
    parser.add_argument('-metric_para', help="set the parameter for different metrics", type=float)
    parser.add_argument('-quantize_test', help="fuzzer for quantization", default=0, type = int)
    parser.add_argument('-quan_model_dir', help="directory including the quantized models for testing")
    parser.add_argument('-random', help="whether to adopt random testing strategy", type=int, default=0)
    parser.add_argument('-select',help="test selection strategy", choices=['uniform', 'tensorfuzz', 'deeptest', 'prob'], default='prob')
    parser.add_argument('-timeout',help="timeout of fuzzing", type=int, default=0)
    parser.add_argument('-clss',help="mutation type of deephunter(Class A = 1, Class A and B = 0", type=int, choices=[0, 1], default=1)
    parser.add_argument('-mutation',help="mutation function", type=str, choices=['deephunter', 'tensorfuzz'], default='deephunter')

    args = parser.parse_args()

    img_rows, img_cols = 224, 224
    input_shape = (img_rows, img_cols, 3)
    input_tensor = Input(shape=input_shape)

    # Get the layers which will be excluded during the coverage computation
    exclude_layer_list = execlude_layer_dic[args.model]

    # Create the output directory including seed queue and crash dir, it is like AFL
    if os.path.exists(args.o):
        shutil.rmtree(args.o)
    os.makedirs(os.path.join(args.o, 'queue'))
    os.makedirs(os.path.join(args.o, 'crashes'))

    # Load model. For ImageNet, we use the default models from Keras framework.
    # For other models, we load the model from the h5 file.
    model = None
    if args.model == 'mobilenet':
        model = MobileNet(input_tensor=input_tensor)
    #elif args.model == 'vgg19':
    #    model = VGG19(input_tensor=input_tensor)
    elif args.model == 'resnet50':
        model = ResNet50(input_tensor=input_tensor)
    else:
        model = load_model(model_weight_path[args.model])

    # Get the preprocess function based on different dataset
    preprocess = preprocess_dic[args.model]

    # Load the profiling information which is needed by the metrics in DeepGauge
    with open(model_profile_path[args.model], 'rb') as rfile:
        u = pickle._Unpickler(rfile)
        u.encoding = 'latin1'
        profile_dict = u.load()
    #profile_dict = pickle.load(open(model_profile_path[args.model], 'rb'), encoding='byte')

    # Load the configuration for the selected metrics.
    if args.metric_para is None:
        cri = metrics_para[args.criteria]
    elif args.criteria == 'nc':
        cri = args.metric_para
    else:
        cri = int(args.metric_para)
    
    # The coverage computer
    coverage_handler = Coverage(model = model, criteria=args.criteria, k = cri,
                                profiling_dict=profile_dict,exclude_layer=exclude_layer_list)
    
    
    # The log file which records the plot data after each iteration of the fuzzing
    plot_file = open(os.path.join(args.o, 'plot.log'), 'a+')

    # If testing for quantization, we will load the quantized versions
    # fetch_function is to perform the prediction and obtain the outputs of each layers
    if args.quantize_test == 1:
        model_names = os.listdir(args.quan_model_dir)
        model_paths = [os.path.join(args.quan_model_dir, name) for name in model_names]
        if args.model == 'mobilenet':
            import keras
            with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,
                                    'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
                models = [load_model(m) for m in model_paths]
        else:
            models = [load_model(m) for m in model_paths]
        fetch_function = build_fetch_function(coverage_handler, preprocess, models)
        model_names.insert(0, args.model)
    else:
        fetch_function = build_fetch_function(coverage_handler, preprocess)
        model_names = [args.model]

    # Like AFL, dry_run will run all initial seeds and keep all initial seeds in the seed queue
    dry_run_fetch = build_fetch_function(coverage_handler, preprocess)



    # The function to update coverage
    coverage_function = coverage_handler.update_coverage
    # The function to perform the mutation from one seed
    mutation_function = image_mutation_function(args.batch_num, args.mutation)

    maxExec = 1
    for exec in range(maxExec):
        # The seed queue
        queue = ImageInputCorpus(args.o,args.random, args.select, coverage_handler.total_size, args.criteria)

        # Perform the dry_run process from the initial seeds
        dry_run(args.model, args.i, dry_run_fetch,coverage_function, queue, args.clss)

        # For each seed, compute the coverage and check whether it is a "bug", i.e., adversarial example
        image_iterate_function = iterate_function(model_names)

        # The main fuzzer class
        fuzzer = Fuzzer(queue, coverage_function, metadata_function,objective_function, mutation_function, fetch_function, image_iterate_function, args.select, args.model, exec)

        # The fuzzing process
        fuzzer.loop(args.max_iteration, args.timeout)

        print('finish', time.time() - start_time)
        fuzzer.saveResult(time.time() - start_time)



