import pickle

model_profile_path = {
    'vgg16': "./profile/cifar10/profiling/vgg16/0_50000.pickle",
    'resnet20': "./profile/cifar10/profiling/resnet20/0_50000.pickle",
    'lenet1': "./profile/mnist/profiling/lenet1/0_60000.pickle",
    'lenet4': "./profile/mnist/profiling/lenet4/0_60000.pickle",
    'lenet5': "./profile/mnist/profiling/lenet5/0_60000.pickle",
    'mobilenet': "./profile/imagenet/profiling/mobilenet_merged.pickle",
    'vgg19': "./profile/imagenet/profiling/vgg19_merged.pickle",
    'resnet50': "./profile/imagenet/profiling/resnet50_merged.pickle"
}

# Load the profiling information which is needed by the metrics in DeepGauge
f = open(model_profile_path['lenet5'], 'rb')
profile_dict = pickle.load(f, encoding='latin1')

print(profile_dict)