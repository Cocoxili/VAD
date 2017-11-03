
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from network import *
from data_process import *
from torchvision import datasets, transforms, models
import time
import os
"""
usage:
    python main.py
    python main.py --network=dnn --mode=test --model='../model/dnn_mix.pkl'

__Author__ = Zhu.Bq

"""

parser = argparse.ArgumentParser(description='pytorch DNN for VAD')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                            help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                            help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=4, metavar='N',
                            help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.008, metavar='LR',
                            help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                            help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
# parser.add_argument('--gpu', type=list, default=[4, 5, 6, 7],
#                             help='gpu device number')
parser.add_argument('--seed', type=int, default=777, metavar='S',
                            help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=5000, metavar='N',
                            help='how many batches to wait before logging training status')
parser.add_argument('--network', type=str, default='dnn',
                            help='densenet or dnn')
parser.add_argument('--mode', type=str, default='train',
                            help='train or test')
parser.add_argument('--model', type=str, default='../model/dnn_2000h_v1_gpu06.pkl',
                            help='trained model path')

os.environ['CUDA_VISIBLE_DEVICES'] = "5"

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

#  torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    #  torch.cuda.set_device(2)


total_correct = 0
total_sample = 0


def train(model, optimizer, train_loader, epoch):
#{{{
    model.train()
    for idx, (data, label) in enumerate(train_loader):

        #  reshape to torch.LongTensor of size 64
        label = label.resize_(label.size()[0])

        if args.cuda:
            data, label = data.cuda(), label.cuda()
        data, label = Variable(data), Variable(label)

        optimizer.zero_grad()

        # print data.size()
        output = model(data)
        # print data.size()

        loss = F.cross_entropy(output, label)

        loss.backward()

        optimizer.step()

        pred = output.data.max(1)[1]  # get the index of the max log-probability
        global total_correct
        total_correct += pred.eq(label.data.view_as(pred)).sum()
        global total_sample
        total_sample += data.size()[0]

        if idx % args.log_interval == 0:

            acc = 100.0 * total_correct / total_sample
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                  'lr:{} Loss: {:.6f}\t'
                  'TrainAcc: {:.2f}'.format(
                    epoch, idx * len(data), len(train_loader.dataset),
                    100. * idx / len(train_loader),
                    optimizer.param_groups[0]['lr'], loss.data[0], acc))

#}}}


def test(model, test_loader):
#{{{
    model.eval()
    test_loss = 0
    correct = 0
    for data, label in test_loader:

        #  reshape to torch.LongTensor of size 64
        label = label.resize_(label.size()[0])

        if args.cuda:
            data, label = data.cuda(), label.cuda()
        data, label = Variable(data, volatile=True), Variable(label)

        output = model(data)

        test_loss += F.cross_entropy(output, label).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(label.data.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

#}}}


def adjust_learning_rate(optimizer, epoch):
    #{{{
    """Sets the learning rate to the initial LR decayed by 2 every 2 epochs"""
    lr = args.lr * (0.5 ** ((epoch-1) // 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr#}}}


def main():
#{{{

    model = DNN()

    if args.cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-6)

    testDataset = FbankDataset('../feature/sampleSet_2000h.test.cPickle', transform=ToTensor())
    test_loader = DataLoader(testDataset, batch_size=args.test_batch_size, shuffle=True, num_workers=8)

    for epoch in range(1, args.epochs + 1):

        start = time.time()

        adjust_learning_rate(optimizer, epoch)

        for tf in range(1, 100):
            trainFile = '../feature/sampleSet_2000h.'+str(tf)+'.cPickle'
            trainDataset = FbankDataset(trainFile, transform=ToTensor())
            train_loader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

            print('training on %s' % trainFile)

            train(model, optimizer, train_loader, epoch)

            test(model, test_loader)

        #  save model every epoch
        model_name = '../model/'+args.network+'_2000h_v2_gpu06.pkl'
        torch.save(model, model_name)
        print('model has been saved as: '+model_name)

        print('time/epoch: %fs' % (time.time() - start))
#}}}


if __name__ == "__main__":

    if args.mode == 'train':
        main()
    if args.mode == 'test':
        model_name = args.model
        model = torch.load(model_name)
        testDataset = FbankDataset('../feature/testHomeNoiseSet.cPickle', transform=ToTensor())
        test_loader = DataLoader(testDataset, batch_size=args.test_batch_size, shuffle=True, num_workers=1)
        test(model, test_loader)
