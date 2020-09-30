"""Pytorch dataset object that loads MNIST dataset as bags."""

import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import pdb
# import torch.utils.data as data
from torch.utils.data import DataLoader
import os
from PIL import Image
import scipy.io as sio
from torchvision.utils import save_image




class MnistBags_outlier_fair(data_utils.Dataset):
    def __init__(self, target_number=9, mean_bag_length=10, var_bag_length=2, num_bag=250, seed=1, train=True):
        self.target_number = target_number
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bag = num_bag
        self.train = train

        self.r = np.random.RandomState(seed)

        self.num_in_train = 60000
        self.num_in_test = 10000

        self.digit = np.array([0,1,2,3,4,5,6,7,8,9])

        if self.train:
            self.train_bags_list, self.train_labels_list = self._create_bags()
        else:
            self.test_bags_list, self.test_labels_list = self._create_bags()

    def _create_bags(self):
        if self.train:
            loader = data_utils.DataLoader(datasets.MNIST('../datasets',
                                                          train=True,
                                                          download=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,), (0.3081,))])),
                                           batch_size=self.num_in_train,
                                           shuffle=False)
        else:
            loader = data_utils.DataLoader(datasets.MNIST('../datasets',
                                                          train=False,
                                                          download=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,), (0.3081,))])),
                                           batch_size=self.num_in_test,
                                           shuffle=False)


        for (batch_data, batch_labels) in loader:
            all_imgs = batch_data
            all_labels = batch_labels

        bags_list = []
        labels_list = []

        for i in range(self.num_bag):
            bag_length = np.int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
            if bag_length < 1:
                bag_length = 1

            target_number = self.digit[i%10]
            select_label = target_number
            self.select_label_ = target_number
            allidx = np.where(all_labels == select_label)[0]
            leftidx = np.setdiff1d(np.arange(all_labels.shape[0]), allidx)


            if self.train:
                indices = torch.LongTensor(self.r.randint(0, self.num_in_train, bag_length))
            else:
                indices = torch.LongTensor(self.r.randint(0, self.num_in_test, bag_length))


            indices = np.where(all_labels == select_label)[0]
            indices = indices[self.r.permutation(indices.shape[0])][0:bag_length]            
            


            labels_in_bag_ = all_labels[indices]
            labels_in_bag = labels_in_bag_ == select_label

            bags_list.append(all_imgs[indices])
            labels_list.append(labels_in_bag)

            labels_in_bag__ = labels_in_bag_.clone()


            
            labels_in_bag__[self.r.permutation(bag_length)[0:2]] = np.setdiff1d(self.digit, select_label)[self.r.permutation(9)[0]]
            for k in range(len(labels_in_bag__)):
              indices[k] = np.where(all_labels == labels_in_bag__[k])[0][self.r.permutation(np.where(all_labels == labels_in_bag__[k])[0].shape[0])][0]

            labels_in_bag___ = labels_in_bag__ == select_label 

            bags_list.append(all_imgs[indices])
            labels_list.append(labels_in_bag___)


        return bags_list, labels_list

    def __len__(self):
        if self.train:
            return len(self.train_labels_list)
        else:
            return len(self.test_labels_list)

    def __getitem__(self, index):
        if self.train:
            bag = self.train_bags_list[index]
            label = [min(self.train_labels_list[index]), self.train_labels_list[index]]
        else:
            bag = self.test_bags_list[index]
            label = [min(self.test_labels_list[index]), self.test_labels_list[index]]


        return bag, label




