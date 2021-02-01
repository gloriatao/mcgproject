import os
import random
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
import pickle
import torch.nn.functional as F

class load_mcg_test(Dataset):
    def __init__(self, imgDirPath, labelDirPath, num_classes, fold='1', shape=None):
        self.shape = shape
        self.imgDirPath = imgDirPath
        self.labelDirPath = labelDirPath
        self.num_classes = num_classes
        self.input_dim = 600
        self.duration_scale = 100  # ms

        # infile = open('./dataset/fold.pickle', 'rb')
        # fold_list = pickle.load(infile)
        # infile.close()
        #
        # if fold == '1':
        #     self.train_list = fold_list['1']
        #     self.test_list = fold_list['2']
        # if fold == '2':
        #     self.test_list = fold_list['1']
        #     self.train_list = fold_list['2']
        self.test_list = ['1', '2']
        self.anno = self.load_label()

        self.nSamples = len(self.test_list)
        print("done init")

    def load_label(self):
        anno = dict()
        csvfile = open(self.labelDirPath, 'r')
        gt = pd.read_csv(csvfile, encoding="utf-8")
        csvfile.close()
        gt = gt.fillna(0)
        Q, R, S, T = gt['Q'].tolist(), gt['R'].tolist(), gt['S'].tolist(), gt['T'].tolist()
        subject = gt['subject']
        ischemia = gt['Ischemia']
        for i in range(len(gt)):
            anno[subject[i]] = {'ischemia':ischemia[i], 'event_class':[Q[i],R[i],S[i],T[i]]}
        return anno

    def __len__(self):
        return self.nSamples

    def generate_heatmap_target(self, heatmap_size, loc, r):
        r = torch.tensor(r)
        aranges = [torch.arange(s) for s in heatmap_size]
        grid = torch.meshgrid(*aranges)
        grid_stacked = torch.stack(grid, axis=2)
        squared_distances = torch.sum(torch.pow(grid_stacked - torch.tensor([loc]), 2.0), axis=-1)
        heatmap = torch.exp(-squared_distances / (2 * torch.pow(r, 2)))
        return heatmap

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        id = self.test_list[index]

        infile = open(os.path.join(self.imgDirPath, id+ '.pickle'), 'rb')
        data = pickle.load(infile)
        infile.close()

        amcg = torch.tensor(data['amcg'])
        amcg = amcg/torch.max(torch.abs(amcg))  ##########
        time_stamp = [data['Q'],data['R'],data['S'],data['T']]
        event_duration = [data['dQ'], data['dR'], data['dS'], data['dT']]
        r =[event_duration[0]//2,event_duration[1]//2,event_duration[2]//2,event_duration[3]//3]
        event_mask = torch.zeros((self.num_classes, amcg.shape[-1]))
        for i in range(self.num_classes):
            if i == 0:
                continue
            else:
                event_mask[i,:] = self.generate_heatmap_target((1,amcg.shape[-1]), [0,time_stamp[i-1]], float(r[i-1]))
        event_mask[0,:] = 1 - torch.sum(event_mask[1:,:],dim=0, keepdim=True)

        amcg = amcg[None,:,:,:]
        is_ischemia = torch.tensor(self.anno[int(id)]['ischemia']).float()
        event_class = torch.tensor(self.anno[int(id)]['event_class']).float()
        event_duration = torch.tensor(event_duration).float()/self.duration_scale
        # label

        return amcg, is_ischemia, event_mask, event_class, event_duration, id


