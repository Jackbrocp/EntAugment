from .ResNet import *
from .resnet2 import *
from .AlexNet import *
from .VGG import *
from .Wide_Resnet import *
from .shakeshake.shake_resnet import *
from .shakeshake.shake_resnext import *
from .pyramidnet import *
from .shake_pyramidnet import *
from .shakedrop import *
def num_class(dataset):
    return {
        'cifar10': 10,
        'reduced_cifar10': 10,
        'cifar10.1': 10,
        'cifar100': 100,
        'svhn': 10,
        'reduced_svhn': 10,
        'imagenet': 1000,
        'reduced_imagenet': 120,
    }[dataset]
def get_model(net_name,dataset='cifar', num_classes=10, local_rank=-1):
    if net_name == 'resnet18':
        model = ResNet18(num_classes=num_classes)
    elif net_name == 'resnet50':
        model = ResNet50(num_classes=num_classes)
    elif net_name == 'resnet101':
        model = ResNet101()
    elif net_name == 'resnet32':
        model = ResNet32()
    elif net_name == 'resnet44':
        model = ResNet44(num_classes)
    elif net_name == 'resnet56':
        model = ResNet56()
    elif net_name == 'resnet110':
        model = ResNet110(num_classes=num_classes)
    elif net_name == 'wresnet28_10':
        model = WideResNet( 28, 10, 0.0, num_classes)
    elif net_name == 'pyramidnet':
        model = PyramidNet('cifar10',200,240,num_classes, True)
    elif net_name == 'pyramidnet-shake':
        model = ShakePyramidNet(depth=110,alpha=270,label=num_classes)
    elif net_name == 'wresnet40_2':
        model = Wide_ResNet( 40, 2, 0.3, num_classes)
    elif net_name == 'shakeshake26_2x32d':
        model = ShakeResNet(26,32,num_classes) 
    elif net_name == 'shakeshake26_2x64d':
        model = ShakeResNet(26,64,num_classes)
    elif net_name == 'shakeshake26_2x96d':
        model =  ShakeResNet(26, 96, num_classes)
    elif net_name =='shakeshake26_2x112d':
        model = ShakeResNet(26,112,num_classes)
    return model

if __name__ =='__main__':
    get_model('resnet32')