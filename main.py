from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable

from dataloader import MnistBags_outlier_fair

from model import LSTM_based_infoLoss

import pdb
import matplotlib

matplotlib.use('Agg')

import os



parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--target_number', type=int, default=9, metavar='T',
                    help='bags have a positive labels if they contain at least one 9')
parser.add_argument('--target_number_2', type=int, default=6, metavar='T',
                    help='bags have a positive labels if they contain at least one 9')
parser.add_argument('--mean_bag_length', type=int, default=6, metavar='ML',
                    help='average bag length')
parser.add_argument('--var_bag_length', type=int, default=1, metavar='VL',
                    help='variance of bag length')
parser.add_argument('--num_bags_train', type=int, default=100, metavar='NTrain',
                    help='number of bags in training set')
parser.add_argument('--num_bags_test', type=int, default=20, metavar='NTest',
                    help='number of bags in test set')
parser.add_argument('--seed', type=int, default=2, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')


all_test_error = np.array([])
all_val_loss = np.array([])



val_all = np.array([])

model_path = './'
if not os.path.exists(model_path):
    os.makedirs(model_path)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

print('Load Train and Test Set')
loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


train_loader = data_utils.DataLoader(MnistBags_outlier_fair(target_number=args.target_number,
                                               mean_bag_length=args.mean_bag_length,
                                               var_bag_length=args.var_bag_length,
                                               num_bag=args.num_bags_train,
                                               seed=args.seed,
                                               train=True),
                                     batch_size=1,
                                     shuffle=True,
                                     **loader_kwargs)


test_loader = data_utils.DataLoader(MnistBags_outlier_fair(target_number=args.target_number,
                                              mean_bag_length=args.mean_bag_length,
                                              var_bag_length=args.var_bag_length,
                                              num_bag=args.num_bags_test,
                                              seed=args.seed,
                                              train=False),
                                    batch_size=1,
                                    shuffle=False,
                                    **loader_kwargs)


val_loader = data_utils.DataLoader(MnistBags_outlier_fair(target_number=args.target_number,
                                              mean_bag_length=args.mean_bag_length,
                                              var_bag_length=args.var_bag_length,
                                              num_bag=args.num_bags_test,
                                              seed=args.seed+1,
                                              train=False),
                                    batch_size=1,
                                    shuffle=False,
                                    **loader_kwargs)

model = LSTM_based_infoLoss()


if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

def train(epoch):
    model.train()
    train_loss = 0.
    train_error = 0.

    for batch_idx, (data, label) in enumerate(train_loader):

  
        bag_label = label[0]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)


        optimizer.zero_grad()

        loss = 5*model.calculate_objective(data, bag_label[0])
        train_loss += loss.data[0]

        error, _ = model.calculate_classification_error(data, bag_label)
        train_error += error
        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader)
    train_error /= len(train_loader)

    print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}, sum:{}'.format(epoch, train_loss.cpu().numpy()[0], train_error, train_loss.cpu().numpy()[0]+train_error))
    
    model.eval()
    val()
    test()



    if epoch == 1:
      model_path_ = '{}/reproduce_LSTM_outlier.pt'.format(model_path)

      model.save_model(model_path_)

    else:
      if val_all[epoch-1] < val_all[epoch-2]:
        model_path_ = '{}/reproduce_LSTM_outlier.pt'.format(model_path)

        model.save_model(model_path_)



def val(showDetail=False):


    global val_all

    
    model.eval()

    test_loss = 0.
    test_error = 0.
    test_error_fake_0 = 0.
    test_error_fake_1 = 0.
    for batch_idx, (data, label) in enumerate(val_loader):
        bag_label = label[0]
        instance_labels = label[1]

        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        loss = model.calculate_objective_val(data, bag_label)
        test_loss += loss.data[0]
        error, predicted_label = model.calculate_classification_error(data, bag_label)
        error_fake_0,  error_fake_1 = model.calculate_classification_error_fake(bag_label)
        test_error += error
        test_error_fake_0 += error_fake_0
        test_error_fake_1 += error_fake_1


    test_error /= len(val_loader)
    test_error_fake_0 /= len(val_loader)
    test_error_fake_1 /= len(val_loader)
    test_loss /= len(val_loader)

    val_all = np.append(val_all, test_loss.cpu().numpy()[0]+test_error)

    print('Val Set, Loss: {:.4f}, Val error: {:.4f}, sum: {:.4f}'.format(test_loss.cpu().numpy()[0], test_error, test_loss.cpu().numpy()[0]+test_error))


def test(showDetail=False):

    
    gt = np.array([])
    pred = np.array([])


    model.eval()

    test_loss = 0.
    test_error = 0.
    test_error_fake_0 = 0.
    test_error_fake_1 = 0.

    for batch_idx, (data, label) in enumerate(test_loader):
        bag_label = label[0]
        instance_labels = label[1]

        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        loss = model.calculate_objective(data, bag_label)
        test_loss += loss.data[0]
        error, predicted_label = model.calculate_classification_error(data, bag_label)
        error_fake_0,  error_fake_1 = model.calculate_classification_error_fake(bag_label)
        test_error += error
        test_error_fake_0 += error_fake_0
        test_error_fake_1 += error_fake_1

        instance_labels = instance_labels[0].numpy()
        gt = np.append(gt, instance_labels)

        predicted_label = predicted_label.cpu().numpy()[0]
        pred = np.append(pred, predicted_label)


    test_error /= len(test_loader)
    test_error_fake_0 /= len(test_loader)
    test_error_fake_1 /= len(test_loader)
    test_loss /= len(test_loader)


    print('Test Set, Loss: {:.4f}, Test error: {:.4f},  Test error fake: {:.4f}, {:.4f}\n'.format(test_loss.cpu().numpy()[0], test_error, test_error_fake_0, test_error_fake_1))

    return test_error



if __name__ == "__main__":

    print('Start Training')
    for epoch in range(1, args.epochs + 1):
        train(epoch)
    print('Start Testing')



