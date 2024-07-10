from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
import pickle
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
from scipy.stats import entropy
import torch.nn.functional as F
from organize_transform import make_transform, make_magnitude_EntAugment
from augmentation.cutout import Cutout
class CIFAR10Dataset(Dataset):
    base_folder=''
    train_list=[
        ['adv_data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['adv_data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['adv_data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['adv_data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['adv_data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
    test_list=[
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }
    transform_list = []

    def __init__(self,root,train=True,transform=None, target_transform=None,aug=None):
        super(CIFAR10Dataset,self).__init__()
        self.train=train
        self.root=root
        self.data: Any = []
        self.targets=[]
        self.filename_list=[]
        self.transform=transform
        self.target_transform=target_transform
        self.make_magnitude_transform = make_magnitude_EntAugment
        if self.train:
            data_list = self.train_list
        else:
            data_list = self.test_list
        
        for file_name, checksum in data_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f,encoding='latin1')
                self.data.append(entry['data'])
                if self.train:
                    self.filename_list.extend(entry['filenames'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  
        self._load_meta()
        self.MAGNITUDE = torch.zeros(50000)
        self.is_warmup_finished = False
        self.warmup_transform = make_transform( length=16)[0]
        
    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
    def set_transform(self, magnitude,cutout_length):
        transform_list = []
        for i in range(len(magnitude)):
            transform_list.append(self.make_magnitude_transform(magnitude=magnitude, cutout_length=cutout_length)) 
        self.transform_list = transform_list
        return 
    def set_MAGNITUDE(self,idx, magnitude):
        self.MAGNITUDE[idx] = magnitude 
    def __getitem__(self, index:int):
        if self.train:
            img, target = self.data[index], int(self.targets[index]) 
        else:
            img, target = self.data[index], int(self.targets[index])
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.train:
            if self.is_warmup_finished:
                t = self.make_magnitude_transform(magnitude=self.MAGNITUDE[index].item(), cutout_length=16)
                normalized_img = t(img)
            else:   
                normalized_img = self.warmup_transform(img)
            return index, normalized_img, target 
        else:
            normalized_img = self.transform(img)
            return normalized_img, target
    def __len__(self):
        return len(self.data)
    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

class CIFAR100Dataset(Dataset):
    def __init__(self, root, train=True, fine_label=True, transform=True, aug=None):
        if train:
            self.data,self.labels,self.filename_list=load_CIFAR_100(root,train,fine_label=fine_label)
        else:
            self.data,self.labels = load_CIFAR_100(root,train,fine_label=fine_label)
        self.transform = transform
        self.train = train
        self.make_magnitude_transform = make_magnitude_EntAugment
        self.MAGNITUDE = torch.zeros(50000)
        self.is_warmup_finished = False
        self.warmup_transform, self.test_transform = make_transform(length=8)
    def set_MAGNITUDE(self, idx, magnitude):
        self.MAGNITUDE[idx] = magnitude 
    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], int(self.labels[index]) 
            if self.is_warmup_finished:
                t = self.make_magnitude_transform(magnitude=self.MAGNITUDE[index].item(), cutout_length=8)
                normalized_img = t(img)
            else:
                normalized_img = self.warmup_transform(img)
            return index, normalized_img, target
        else:
            img, target = self.data[index], int(self.labels[index])
            normalized_img = self.test_transform(img)
            return normalized_img, target
    def __len__(self):
        return len(self.data)

def load_CIFAR_100(root, train=True, fine_label=True):
    if train:
        filename = root + 'my_train'
    else:
        filename = root + 'test'
 
    with open(filename, 'rb')as f:
        datadict = pickle.load(f,encoding='bytes')
 
        if train:
            # [50000, 32, 32, 3]
            X = datadict['data']
            filename_list = datadict['filenames']
            X = X.reshape(50000, 3, 32, 32).transpose(0,2,3,1)
            Y = datadict['labels']
            Y = np.array(Y)
            return X, Y, filename_list
        else:
            # [10000, 32, 32, 3]
            X = datadict[b'data']
            filename_list = datadict[b'filenames']
            X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1)
            Y = datadict[b'fine_labels']
            Y = np.array(Y)
            return X, Y
