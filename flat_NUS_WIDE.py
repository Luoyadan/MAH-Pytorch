import pickle
import os
import argparse
import logging
import torch
import time

import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms

from datetime import datetime
from torch.autograd import Variable
from torch.utils.data import DataLoader
import sys
sys.path.append('./utils')

import torch.nn as nn
import data_processing as dp
import adsh_loss as al
import cnn_model as cnn_model
import subset_sampler as subsetsampler
import calc_hr as calc_hr
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description="ADSH demo")
parser.add_argument('--bits', default='4,8,12', type=str,
                    help='binary code length (default: 12,24,32,48)')
parser.add_argument('--gpu', default='0', type=str,
                    help='selected gpu (default: 1)')
parser.add_argument('--arch', default='multihead', type=str,
                    help='model name (default: resnet50)')
parser.add_argument('--max-iter', default=300, type=int,
                    help='maximum iteration (default: 50)')
parser.add_argument('--epochs', default=1, type=int,
                    help='number of epochs (default: 3)')
parser.add_argument('--batch-size', default=100, type=int,
                    help='batch size (default: 64)')

parser.add_argument('--num-samples', default=8000, type=int,
                    help='hyper-parameter: number of samples (default: 2000)')
parser.add_argument('--gamma', default=200, type=int,
                    help='hyper-parameter: gamma (default: 200)')
parser.add_argument('--learning-rate', default=0.001, type=float,
                    help='hyper-parameter: learning rate (default: 10**-3)')


def _logging():
    os.mkdir(logdir)
    global logger
    logfile = os.path.join(logdir, 'log.log')
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    _format = logging.Formatter("%(name)-4s: %(levelname)-4s: %(message)s")
    fh.setFormatter(_format)
    ch.setFormatter(_format)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return

def _record():
    global record
    record = {}
    record['train loss'] = []
    record['iter time'] = []
    record['qB'] = []
    record['rB'] = []
    record['pre'] = []
    record['rec'] = []
    record['map'] = []
    record['topkpre'] = []
    record['param'] = {}
    return

def _save_record(record, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(record, fp)
    return


def encoding_onehot(target, nclasses=10):
    target_onehot = torch.FloatTensor(target.size(0), nclasses)
    target_onehot.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1)
    return target_onehot


def _dataset():
    normalize = transforms.Normalize(mean=[0.38933693 , 0.42026136,  0.43608778], std=[0.23933473,  0.23539561,  0.24641008])
    transformations = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    dset_database = dp.DatasetProcessingNUS_WIDE(
        'data/NUS-WIDE', 'database_image_new.txt', 'database_label_new.txt', transformations
    )
    dset_test = dp.DatasetProcessingNUS_WIDE(
        'data/NUS-WIDE', 'test_image_new.txt', 'test_label_new.txt', transformations
    )
    num_database, num_test = len(dset_database), len(dset_test)

    def load_label(filename, DATA_DIR):
        label_filepath = os.path.join(DATA_DIR, filename)
        label = np.loadtxt(label_filepath, dtype=np.int64)
        return torch.from_numpy(label)

    databaselabels = load_label('database_label_new.txt', 'data/NUS-WIDE')
    testlabels = load_label('test_label_new.txt', 'data/NUS-WIDE')

    dsets = (dset_database, dset_test)
    nums = (num_database, num_test)
    labels = (databaselabels, testlabels)
    return nums, dsets, labels


def calc_sim(database_label, train_label):
    S = (database_label.mm(train_label.t()) > 0).type(torch.FloatTensor)
    '''
    soft constraint
    '''
    r = S.sum() / (1-S).sum()
    S = S*(1+r) - r
    return S

def calc_loss(V, U, S, code_length, select_index, gamma):
    num_database = V.shape[0]
    square_loss = (U.dot(V.transpose()) - code_length*S) ** 2
    V_omega = V[select_index, :]
    quantization_loss = (U-V_omega) ** 2
    loss = (square_loss.sum() + gamma * quantization_loss.sum()) / (opt.num_samples * num_database)
    return loss

def encode(model, data_loader, num_data, bits):
    bit, bit1, bit2 = bits
    B = np.zeros([num_data, bit], dtype=np.float32)
    B1 = np.zeros([num_data, bit1], dtype=np.float32)
    B2 = np.zeros([num_data, bit2], dtype=np.float32)
    for iter, data in enumerate(data_loader, 0):
        data_input, _, data_ind = data
        data_input = Variable(data_input.cuda())
        output, output1, output2 = model(data_input)
        B[data_ind.numpy(), :] = torch.sign(output.cpu().data).numpy()
        B1[data_ind.numpy(), :] = torch.sign(output1.cpu().data).numpy()
        B2[data_ind.numpy(), :] = torch.sign(output2.cpu().data).numpy()
    return B, B1, B2

def adjusting_learning_rate(optimizer, iter):
    update_list = [10, 20, 30, 40, 50]
    if iter in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 10

def adsh_algo(code_lengths):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    code_length, code_length1, code_length2 = code_lengths
    '''
    parameter setting
    '''
    max_iter = opt.max_iter
    epochs = opt.epochs
    batch_size = opt.batch_size
    learning_rate = opt.learning_rate
    weight_decay = 5 * 10 ** -4
    num_samples = opt.num_samples
    gamma = opt.gamma

    record['param']['topk'] = 5000
    record['param']['opt'] = opt
    record['param']['description'] = '[Comment: learning rate decay]'
    logger.info(opt)
    logger.info(code_length)
    logger.info(record['param']['description'])

    '''
    dataset preprocessing
    '''
    nums, dsets, labels = _dataset()
    num_database, num_test = nums
    dset_database, dset_test = dsets
    database_labels, test_labels = labels

    '''
    model construction
    '''
    model = cnn_model.CNNNet(opt.arch, code_lengths)

    model = nn.DataParallel(model)
    model.cuda()
    adsh_loss = al.ADSHLoss(gamma, code_length, num_database)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    V = np.zeros((num_database, code_length))
    V1 = np.zeros((num_database, code_length1))
    V2 = np.zeros((num_database, code_length2))


    for iter in range(max_iter):
        model.train()
        iter_time = time.time()
        '''
        sampling and construct similarity matrix
        '''
        select_index = list(np.random.permutation(range(num_database)))[0: num_samples]
        _sampler = subsetsampler.SubsetSampler(select_index)
        trainloader = DataLoader(dset_database, batch_size=batch_size,
                                 sampler=_sampler,
                                 shuffle=False,
                                 num_workers=4)
        '''
        learning deep neural network: feature learning
        '''
        sample_label = database_labels.index_select(0, torch.from_numpy(np.array(select_index)))
        Sim = calc_sim(sample_label, database_labels)
        U = np.zeros((num_samples, code_length), dtype=np.float)
        U1 = np.zeros((num_samples, code_length1), dtype=np.float)
        U2 = np.zeros((num_samples, code_length2), dtype=np.float)
        for epoch in range(epochs):
            for iteration, (train_input, train_label, batch_ind) in enumerate(trainloader):
                batch_size_ = train_label.size(0)
                u_ind = np.linspace(iteration * batch_size, np.min((num_samples, (iteration+1)*batch_size)) - 1, batch_size_, dtype=int)
                train_input = Variable(train_input.cuda())
                #print(train_input.shape)
                code, code1, code2 = model(train_input)
                S = Sim.index_select(0, torch.from_numpy(u_ind))
                U[u_ind, :] = code.cpu().data.numpy()
                U1[u_ind, :] = code1.cpu().data.numpy()
                U2[u_ind, :] = code2.cpu().data.numpy()

                model.zero_grad()
                loss0 = 10 * adsh_loss(code, code_length, V, S, V[batch_ind.cpu().numpy(), :])
                loss1 = 2 * adsh_loss(code1, code_length1, V1, S, V1[batch_ind.cpu().numpy(), :])
                loss2 = adsh_loss(code2, code_length2, V2, S, V2[batch_ind.cpu().numpy(), :])
                loss = loss0 + loss1 + loss2
                #if iteration % 50 == 0:
                #    logger.info('[Iteration: %3d/%3d][Train Loss #1: %.4f][Train Loss #2: %.4f][Train Loss #3: %.4f]', iter, max_iter, loss0, loss1, loss2)
                loss.backward()
                optimizer.step()
        adjusting_learning_rate(optimizer, iter)

        '''
        learning binary codes: discrete coding
        '''
        barU = np.zeros((num_database, code_length))
        barU[select_index, :] = U
        Q = -2*code_length*Sim.cpu().numpy().transpose().dot(U) - 2 * gamma * barU
        for k in range(code_length):
            sel_ind = np.setdiff1d([ii for ii in range(code_length)], k)
            V_ = V[:, sel_ind]
            Uk = U[:, k]
            U_ = U[:, sel_ind]
            V[:, k] = -np.sign(Q[:, k] + 2 * V_.dot(U_.transpose().dot(Uk)))

        barU1 = np.zeros((num_database, code_length1))
        barU1[select_index, :] = U1
        Q1 = -2 * code_length1* Sim.cpu().numpy().transpose().dot(U1) - 2 * gamma * barU1
        for k in range(code_length1):
            sel_ind = np.setdiff1d([ii for ii in range(code_length1)], k)
            V1_ = V1[:, sel_ind]
            U1k = U1[:, k]
            U1_ = U1[:, sel_ind]
            V1[:, k] = -np.sign(Q1[:, k] + 2 * V1_.dot(U1_.transpose().dot(U1k)))

        barU2 = np.zeros((num_database, code_length2))
        barU2[select_index, :] = U2
        Q2 = -2 * code_length2 * Sim.cpu().numpy().transpose().dot(U2) - 2 * gamma * barU2
        for k in range(code_length2):
            sel_ind = np.setdiff1d([ii for ii in range(code_length2)], k)
            V2_ = V2[:, sel_ind]
            U2k = U2[:, k]
            U2_ = U2[:, sel_ind]
            V2[:, k] = -np.sign(Q2[:, k] + 2 * V2_.dot(U2_.transpose().dot(U2k)))
        iter_time = time.time() - iter_time
        loss_ = calc_loss(V, U, Sim.cpu().numpy(), code_length, select_index, gamma)
        loss1_ = calc_loss(V1, U1, Sim.cpu().numpy(), code_length1, select_index, gamma)
        loss2_ = calc_loss(V2, U2, Sim.cpu().numpy(), code_length2, select_index, gamma)
        logger.info('[Iteration: %3d/%3d][After Optimization Train Loss: %.4f][2# Train Loss: %.4f][3# Train Loss: %.4f]', iter, max_iter, loss_, loss1_, loss2_)
        record['train loss'].append(loss_)
        record['iter time'].append(iter_time)

        '''
        training procedure finishes, evaluation
        '''
        if iter % 10 == 0:
            model.eval()
            testloader = DataLoader(dset_test, batch_size=1,
                             shuffle=False,
                             num_workers=4)
            qB, qB1, qB2 = encode(model, testloader, num_test, code_lengths)
            rB, rB1, rB2 = V, V1, V2
            map = calc_hr.calc_map(qB, rB, test_labels.numpy(), database_labels.numpy())
            map1 = calc_hr.calc_map(qB1, rB1, test_labels.numpy(), database_labels.numpy())
            map2 = calc_hr.calc_map(qB2, rB2, test_labels.numpy(), database_labels.numpy())
            pre, rec = calc_hr.calc_prerec(qB, rB, test_labels.numpy(), database_labels.numpy())
            topkpre = calc_hr.calc_topMap(qB, rB, test_labels.numpy(), database_labels.numpy(), 5000)
            topkpre1 = calc_hr.calc_topMap(qB1, rB1, test_labels.numpy(), database_labels.numpy(), 5000)
            topkpre2 = calc_hr.calc_topMap(qB2, rB2, test_labels.numpy(), database_labels.numpy(), 5000)
            logger.info('[Evaluation: mAP: %.4f]', map)
            logger.info('[Evaluation: TopK@Pre: %.4f]', topkpre)
            logger.info('[2# Evaluation: mAP: %.4f]', map1)
            logger.info('[2# Evaluation: TopK@Pre: %.4f]', topkpre1)
            logger.info('[3# Evaluation: mAP: %.4f]', map2)
            logger.info('[3# Evaluation: TopK@Pre: %.4f]', topkpre2)
            record['rB'].append(rB)
            record['qB'].append(qB)
            record['pre'].append(pre)
            record['rec'].append(rec)

            record['map'].append(map)
            record['map'].append(map1)
            record['map'].append(map2)

            record['topkpre'].append(map)
            record['topkpre'].append(map1)
            record['topkpre'].append(map2)
            filename = os.path.join(logdir, str(code_length) + 'bits-record.pkl')

    _save_record(record, filename)


if __name__=="__main__":
    global opt, logdir
    opt = parser.parse_args()
    logdir = '-'.join(['log/log-ADSH-nuswide', datetime.now().strftime("%y-%m-%d-%H-%M-%S")])
    _logging()
    _record()
    bits = [int(bit) for bit in opt.bits.split(',')]
    adsh_algo(bits)
