# -*- coding: utf-8 -*-

from __future__ import division
from util import *
import os

class Stat():
    def __init__(self, filename):
        self.ali = open(filename)
        self.ali = self.ali.readlines()
        self.amount = {}

    def __len__(self):
        return len(self.ali)

    def __getitem__(self, index):
        return self.ali[index]

    def stat_on_sentence(self):
        #{{{
        """
        计算每一句话中0、1的数量和比例，返回一个二维列表，
        每行代表一句话的统计信息：[key、0的个数、1的个数、0的比例、1的比例]
        stat_on_sentence:
        [
            [key,number_of_zeros,number_of_ones,ratio_of_zeros,ratio_of_ones],[...],...
        ]
        """
        stat_on_sentence = []
        for line in self.ali:
            #  print line
            line =line.strip('\n').split(' ')
            key = line[0]
            labels = line[1:]
            ss = [key, 0, 0, 0, 0]
            for c in labels:
                if c == '0':
                    ss[1] += 1
                elif c == '1':
                    ss[2] += 1
                else:
                    print 'unknow character: ' + c
            ss[3] = round(ss[1]/(ss[1]+ss[2]), 4)
            ss[4] = round(ss[2]/(ss[1]+ss[2]), 4)
            stat_on_sentence.append(ss)
        return stat_on_sentence
#}}}

    def stat_on_dataset(self):
        #{{{
        """
        返回整个数据集标签中0的个数、1的个数、0的比例、1的比例
        """
        stat_on_sentence = self.stat_on_sentence()
        zeros = 0
        ones = 0
        ratio_0 = 0
        ratio_1 = 0
        for ss in stat_on_sentence:
            zeros += ss[1]
            ones += ss[2]
        ratio_0 = round(zeros/(zeros + ones), 4)
        ratio_1 = round(ones/(zeros + ones), 4)
        #  print "total frames: ", zeros+ones
        return (zeros, ones, ratio_0, ratio_1)
#}}}

    def pre_post_zeros(self):
#{{{
        """
        计算每一话中前0和后0的数量与比例，返回一个二维列表，
        每行代表一句话的统计信息：[key、总帧数、前0的个数、后0的个数、前0的比例、后0的比例]
        """
        pre_post_zeros = []
        for line in self.ali:
            #  print line
            line =line.strip('\n').split(' ')
            key = line[0]
            labels = line[1:]
            ss = [key, len(labels), 0, 0, 0, 0]
            for c in labels:
                if c == '0':
                    ss[2] += 1
                elif c == '1':
                    break

            labels.reverse()
            for c in labels:
                if c == '0':
                    ss[3] += 1
                elif c == '1':
                    break
            ss[4] = round(ss[2]/ss[1], 4)
            ss[5] = round(ss[3]/ss[1], 4)
            pre_post_zeros.append(ss)
        return pre_post_zeros
        #}}}


def task1():
#{{{
    """
    静音比例与句子个数之间的关系
    """
    stat = Stat('../data_fbank/ali.phone.sub.txt')
    stat_on_sentence = stat.stat_on_sentence()
    num_sentences = [0] * 100
    #  count = 0
    for ss in stat_on_sentence:
        location = int(ss[3]*100//1)
        num_sentences[location] += 1
        if ss[3]==0:
            count += 1
    print num_sentences
    #  print "without slience: ", count
#}}}

def task2():
    #{{{
    """
    前0、后0比例与句子个数之间的关系
    """
    stat = Stat('../data_fbank/ali.phone.sub.txt')
    pre_post_zeros = stat.pre_post_zeros()
    num_sentences_0 = [0] * 100
    num_sentences_1 = [0] * 100

    count = 0
    frames = 0
    for ss in pre_post_zeros:
        location_0 = int(ss[4]*100//1)
        location_1 = int(ss[5]*100//1)
        num_sentences_0[location_0] += 1
        num_sentences_1[location_1] += 1
        frames += ss[1]
        if ss[2]==0 and ss[3]==0:
            count += 1

        #  exit(0)
    print num_sentences_0
    print num_sentences_1
    print "without pre and post slience: ",count
    print "total frames 2: ", frames
#}}}

def task3():
    """
    statistics on /feature cPickle
    """
    fs = os.listdir('../featureMixNoise')
    fs_abs = []

    for f in fs:
        fs = os.path.join('/home/zbq/work/vad/featureMixNoise/', f)
        fs_abs.append(fs)

    zeros = 0
    ones = 0

    for f in fs_abs:
        print f
        f = load_data(f)
        for sample in f:
            if sample['label'] == 0:
                zeros += 1
            elif sample['label'] == 1:
                ones += 1

    total = zeros + ones
    print "zeros: ", zeros/total
    print "ones: ", ones/total




if __name__ == "__main__":
    #  stat = Stat('../data_fbank/ali.phone.sub.txt')
    #  t = stat.stat_on_dataset()
    #  print t
    #  task1()
    #  task2()
    task3()
