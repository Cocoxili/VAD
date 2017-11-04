# -*- coding: utf-8 -*-

import pandas as pd
import os
from util import *
import numpy
from multiprocessing.dummy import Pool as ThreadPool
"""
Do data transform.
__Author__ = Zhu.Bq

"""


def get_fb(file_name):
    """#{{{
    load feature file and transform to dict

    return:
        dict
        key_list_feat

    """
    ff = open(file_name, 'r')
    fb = []
    delta = []
    fb_matrix = numpy.zeros([1, 24])
    delta_matrix = numpy.zeros([1, 24])
    fbanks = {}
    deltas = {}
    fb_keylist = []

    while(1):
        line = ff.readline()
        if not line:
            #  print 'end of file'
            break
        end_line = line.strip().split()[-1]
        if end_line == '[':
            key = line.strip().split()[0]
        elif end_line == ']':
            for i in range(24):
                fb.append(float(line.strip().split()[i]))
            for i in range(24, 48):
                delta.append(float(line.strip().split()[i]))

            fb_keylist.append(key)

            fb_matrix = numpy.vstack((fb_matrix, fb))
            fbanks[key] = fb_matrix[1:, :]

            delta_matrix = numpy.vstack((delta_matrix, delta))
            deltas[key] = delta_matrix[1:, :]

            fb = []
            delta = []
            fb_matrix = numpy.zeros([1, 24])
            delta_matrix = numpy.zeros([1, 24])

        else:
            for i in range(24):
                #  value.append(line.strip().split()[i])
                fb.append(float(line.strip().split()[i]))
            for i in range(24, 48):
                delta.append(float(line.strip().split()[i]))

            fb_matrix = numpy.vstack((fb_matrix, fb))
            delta_matrix = numpy.vstack((delta_matrix, delta))
            fb = []
            delta = []

    print('number of utterances in fbank: %d' % len(fbanks))
    ff.close()
    return fbanks, deltas, fb_keylist
#}}}


def get_ali(filename):
    """#{{{
    load label file and transform to dict

    """
    lf = open(filename, 'r')
    labels = {}
    ali_keylist = []
    value = []

    while(1):
        line = lf.readline()

        if not line:
            #  print 'end of file'
            break
        line_split = line.strip().split()
        key = line_split[0]
        ali_keylist.append(key)

        for i in range(1, len(line_split)):
             value.append(int(line_split[i]))

        labels[key] = value
        value = []

    print('number of utterances in ali: %d' % len(labels))
    lf.close()

    return labels, ali_keylist#}}}


def get_noise_list(filename):
    #{{{
    file1 = open(filename)
    lines = file1.readlines()
    keys = []
    for line in lines:
        line = line.split(' ')[0]
        keys.append(line)
    return keys#}}}


def generate_train_and_test():
#{{{
    print('../data_fbank --> ../feature')
    #  nf = open('data/meta.txt', 'w')

    label_file = '../ali.phone.txt'

    labels, ali_keylist = get_ali(label_file)


    key_list_noise = get_noise_list('../data_fbank/feats.noise.scp')

    for i in range(19, 16, -1):

        sampleSet = []

        feat_file = '../data_fbank/raw_feat.' + str(i) + '.txt'
        print(feat_file)
        fbanks, deltas, fb_keylist = get_fb(feat_file)
        print("number of fbanks: ", len(fb_keylist))

        for key in fb_keylist:

            if washKey(key) in ali_keylist:
                num_frames = len(labels[washKey(key)])
                if fbanks[key].shape[0] != num_frames:
                    print(key, fbanks[key].shape[0], num_frames)
                num_min = min(fbanks[key].shape[0], num_frames)

                for n in range(num_min):
                    label = labels[washKey(key)][n]

                    sample = {'key': key, 'fb': fbanks[key][n], 'delta': deltas[key][n], 'label': label}

                    sampleSet.append(sample)

            elif key in key_list_noise:
                num_frames = fbanks[key].shape[0]
                for n in range(num_frames):
                    label = 0

                    sample = {'key': key, 'fb': fbanks[key][n], 'delta': deltas[key][n], 'label': label}

                    sampleSet.append(sample)

        if i == 0:
            # testing set
            save_data('../feature/sampleSet_2000h.test.cPickle', sampleSet)
            print('finished testing set.')
        else:
            # training set
            save_data('../feature/sampleSet_2000h.' + str(i) + '.cPickle', sampleSet)

    print('finished training set.')
#}}}


def generate_noise():
    #{{{
    """

    """
    print('generate noise test.')

    #  label_file = '../data_fbank/ali.phone.sub.txt'

    #  labels, key_list_label = get_label_dict(label_file)

    sampleSet = []

    feat_file = '../data_fbank/raw_feat_homeNoise.txt'
    print(feat_file)
    fbanks, deltas, fb_keylist = get_fb(feat_file)

    for key in fb_keylist:
        #  write key in the meta file
        #  nf.write(key+'\n')
        #  if key in key_list_label:
        num_frames = fbanks[key].shape[0]
            #  print len(fbank[key])/24
            #  assert fbank[key].shape[0] == num_frames
        for n in range(num_frames):
            label = 0

            sample = {'key': key, 'fb': fbanks[key][n], 'delta': deltas[key][n], 'label': label}

            sampleSet.append(sample)

    sampleSet = sampleSet[:50000]
    save_data('../feature/testNoiseSet.cPickle', sampleSet)
    print('finished.')
#}}}


def washKey(key):
    #  print key#{{{
    pre = key.split('_')[0]
    #  post = key.split('.')[-1]
    key = pre
    return key#}}}


if __name__ == "__main__":

    generate_train_and_test()
    #  generate_noise()

    # sampleSet = load_data('../feature/sampleSet_2000h.101.cPickle')
    # print len(sampleSet)
    # print len(sampleSet[0]['fb'])
    # print sampleSet[0]

    # for i in range(100):
    #     print sampleSet[i]
