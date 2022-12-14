import os
import random

import cv2 as cv
import numpy as np
import pandas as pd
import torch
from torch.nn import init


def weight_init_kaiming(m):
    class_names = m.__class__.__name__
    if class_names.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif class_names.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)


def box(img, mask):
    if len(img.shape) == 2:
        new_img = np.zeros((img.shape[0], img.shape[1], 3))
        for i in range(3):
            new_img[:, :, i] = img
        img = new_img
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    maskimg = cv.bitwise_and(src1=img, src2=img, mask=mask)
    # print(maskimg.shape)
    sub_img = np.where(maskimg > 0)
    final = maskimg[min(sub_img[0]):max(sub_img[0]), min(sub_img[1]):max(sub_img[1]), :]
    return final


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Log:
    def __init__(self, path, model_name, df_style=None):
        self.path = path
        self.model_name = model_name
        if df_style is None:
            self.df_style = pd.DataFrame(
                columns=['epoch', 'train_loss', 'train_acc', 'train_precision', 'train_recall', 'train_f1',
                         'train_kappa',
                         'val_loss', 'val_acc', 'val_precision', 'val_recall', 'val_f1', 'val_kappa',
                         'test_acc', 'test_precision', 'test_recall', 'test_f1', 'test_kappa'],
                index=[i for i in range(3000)],
            )
        else:
            self.df_style = df_style

        # print(self.df_style)

    def load(self, path=None):
        if path is None:
            self.df_style = pd.read_csv(self.path, index_col=-1)

        else:
            self.df_style = pd.read_csv(path, index_col=-1)

        print('load succeed!')

    def print(self):
        print(self.df_style)

    def record(self, epoch, evaluator, loss, state, auto_write=False):
        r = list(evaluator.analysis())
        self.df_style.iloc[epoch]['epoch'] = epoch
        if state != 'test':
            self.df_style.iloc[epoch][state + '_loss'] = loss
        self.df_style.iloc[epoch][state + '_acc'] = r[0]
        self.df_style.iloc[epoch][state + '_precision'] = r[1]
        self.df_style.iloc[epoch][state + '_recall'] = r[2]
        self.df_style.iloc[epoch][state + '_f1'] = r[3]
        self.df_style.iloc[epoch][state + '_kappa'] = r[4]
        if auto_write:
            self.write()
        else:
            print(
                'The record has been updated, but not yet written to the file because the argument \'auto_write\' is False')

    def write(self):
        self.df_style.to_csv(self.path, index=False)
        print('log succeed!')


def count_parameters(model):
    # ???????????????????????????
    temp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nThe model has {temp:,} trainable parameters')


def get_device():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


def score(state, evaluater, epoch=-1, loss=-1.):
    acc, recall, pre, f1, kappa = evaluater.analysis()
    if state == 'test':
        print('|acc       : {}%'.format(acc * 100))
        print('|recall    : {}'.format(recall))
        print('|precision : {}'.format(pre))
        print('|f1        : {}'.format(f1))
        print('|kappa     : {}'.format(kappa))
    else:
        print('|epoch     : {}    '.format(epoch) + state + ' loss: {}'.format(loss))
        print('|acc       : {}%'.format(acc * 100))
        print('|recall    : {}'.format(recall))
        print('|precision : {}'.format(pre))
        print('|f1        : {}'.format(f1))
        print('|kappa     : {}'.format(kappa))
