# -*- coding: utf-8 -*-

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import librosa
import time
from util import *
"""
optimize I/O with packet sequence input.
__Author__ = Zhu.Bq

"""

os.environ['CUDA_VISIBLE_DEVICES'] = "6"


class FbankDataset(Dataset):
#{{{
    """
    Prepare data for MLP.

    """
    def __init__(self, pkl_file, transform=None, context=True):
        """
        context:
            if true, joint some frame feature as context.
            if false, return current frame feature.
        sampleSet:
            [{'feat':feat,'label':label},{}....{}]

        """
        self.transform = transform
        self.sampleSet = load_data(pkl_file)
        self.context = context

    def __len__(self):
        return len(self.sampleSet)

    def __getitem__(self, index):
        """
        feat:
        label: int

        """
        if self.context:
            feat = self.joint(index)
            label = self.sampleSet[index]['label']
            sample = {'feat': feat, 'label': label}
        else:
            sample = self.sampleSet[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def getFrame(self, index):
        """
        if frame in the context is not exist, copy the
        fisrt of last frame
        realized in recurision method

        """
        if index < 0:
            return self.getFrame(index+1)

        if index >= len(self.sampleSet):
            return self.getFrame(index-1)

        fb = self.sampleSet[index]['fb']
        delta = self.sampleSet[index]['delta']
        f = np.concatenate((fb, delta))
        return f

    def joint(self, current_index):
        feat = []
        #  ac = []
        for i in range(-8, 9):
            feat.append(self.getFrame(current_index+i))

        feat = np.array(feat)
        feat = feat.reshape(-1)
        return feat
#}}}


class ConvDataset(Dataset):
    """#{{{
    Prepare data for Densenet.

    """
    def __init__(self, pkl_file, transform=None):
        """
        sampleSet:
            [{'feat':feat,'label':label},{}....{}]

        """
        self.transform = transform
        self.sampleSet = load_data(pkl_file)
        #  self.pointer = load_data(pointer)

    def __len__(self):
        return len(self.sampleSet)

    def __getitem__(self, index):
        """
        feat: 1 x 33 x 24 nparray
            1 channels, 33rows(11fbanks+11delta+11accelerate), 24 fbanks
        label: int

        """

        feat = self.joint(index)
        # (33, 24) --> (1, 33, 24)
        feat = feat.reshape(1, feat.shape[0], feat.shape[1])
        label = self.sampleSet[index]['label']
        sample = {'feat':feat, 'label':label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def getFrame(self, index):
        """
        if frame in the context is not exit, copy the
        fisrt of last frame
        realized in recurision method

        """
        if index < 0:
            return self.getFrame(index+1)

        if index >= len(self.sampleSet):
            return self.getFrame(index-1)

        f = self.sampleSet[index]['feat']
        return f

    def joint(self, current_index):
        origin = []
        delta = []
        ac = []
        for i in range(-5, 6):
            origin.append(self.getFrame(current_index+i)[0])
            delta.append(self.getFrame(current_index+i)[1])
            ac.append(self.getFrame(current_index+i)[2])

        return np.vstack((origin, delta, ac))#}}}


class LSTMDataset(Dataset):
#{{{
    """

    """
    def __init__(self, pkl_file, transform=None):
        """
        context:
            if true, joint some frame feature as context.
            if false, return current frame feature.
        sampleSet:
            [{'feat':feat,'label':label},{}....{}]

        """
        self.transform = transform
        self.sampleSet = load_data(pkl_file)
        self.delay = 5

    def __len__(self):
        return len(self.sampleSet)

    def __getitem__(self, index):
        """
        feat:
        label: int

        """
        feat = self.sampleSet[index]['feat']
        # feat = np.concatenate((feat[0], feat[1]))
        feat = feat[0]
        if index < 5:
            label = self.sampleSet[index]['label']
        else:
            label = self.sampleSet[index - self.delay]['label']
        sample = {'feat': feat, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
#}}}



class ToTensor(object):
    """#{{{
    convert ndarrays in sample to Tensors.
„
    return:
        feat(torch.FloatTensor)
        label(torch.LongTensor of size batch_size x 1)

    """

    def __call__(self, sample):
        feat, label = sample['feat'], sample['label']

        feat = torch.from_numpy(feat).type(torch.FloatTensor)
        label = torch.LongTensor([label])
        #  feat = torch.rand(1, 33, 34).type(torch.FloatTensor)
        #  label = torch.LongTensor([1])
        return feat, label#}}}


if __name__ == "__main__":
    #  ConvDataset = ConvDataset('../data/sampleSet.cpickle', transform=ToTensor())
    #  dataloader = DataLoader(ConvDataset, batch_size=4, shuffle=False, num_workers=1)

    fbankDataset = FbankDataset('../feature/sampleSet_2000h.101.cPickle', transform=ToTensor())
    dataloader = DataLoader(fbankDataset, batch_size=10, shuffle=False, num_workers=1)

    # RawDataset = RawDataset('../feature/key_wave_dev', '../feature/key_ali_dev', '../feature/index_dev', transform=ToTensor())
    # dataloader = DataLoader(RawDataset, batch_size=2, shuffle=False, num_workers=1)

    # TestDataset = TestDataset(transform=ToTensor())
    # dataloader = DataLoader(TestDataset, batch_size=1, shuffle=False, num_workers=1)
    #  print(fbankDataset[0])

    print len(fbankDataset)

    for idx, sample_batched in enumerate(dataloader):
        print(idx)
        print(sample_batched[0].size())
        # print(type(sample_batched[0].size()))
        if idx == 10:
            break



