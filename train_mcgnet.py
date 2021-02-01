# encoding: utf-8
from __future__ import print_function
import torch.optim as optim
from torch.utils.data import DataLoader
from models import mcg_bert, Criterion3, Postprocess3
import time, logging, os
from config import Config
from utils import *
from dataset import load_mcg_train, load_mcg_test
import torch
import torch.nn as nn
import wandb
# wandb.init(project="GANsyn")
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

def adjust_learning_rate(optimizer, batch, steps, scales, lr):
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch >= steps[i]:
            lr = lr * scale
            if batch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr / batch_size
    return lr

def train_epoch(epoch, train_loader, config, device, writer=None):
    global processed_batches
    model.train()
    processed_batches = 0
    for batch_idx, (amcg, is_ischemia, event_mask, event_class, event_duration) in enumerate(train_loader):
        processed_batches = processed_batches + 1

        out_event, out_event_cls, out_event_duration,_ = model(amcg.cuda())  # 1, 1152, 150
        loss_dict = Criterion3(out_event, out_event_cls, out_event_duration, is_ischemia.cuda(), event_mask.cuda(), event_class.cuda(), event_duration.cuda())

        # Losses
        weight_dict = {'loss_event_dice':1,'loss_event_mse':10,'loss_event_cls':1, 'loss_duration':1}
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # accuracy
        print(' %d,  %d, loss: %f,  loss_event_dice: %f, loss_event_mse: %f, loss_event_cls: %f, loss_event_duration: %f'
              % (epoch, processed_batches, losses.item(),
                 loss_dict['loss_event_dice'].item(), loss_dict['loss_event_mse'].item(), loss_dict['loss_event_cls'].item(), loss_dict['loss_duration'].item(), ))

    return

def evaluate_epoch(valid_loader, model):
    model.eval()
    print('-----------Start Validation------------')
    with torch.no_grad():
        losses_test = 0
        cnt = 0
        pred_event_cls=[]
        gt_event_cls=[]
        gt_event_duration=[]
        for batch_idx, (amcg, is_ischemia, event_mask, event_class,event_duration, id) in enumerate(valid_loader):
            cnt += amcg.shape[0]

            out_event, out_event_cls, out_event_duration, _ = model(amcg.cuda(), mode='test')  # 1, 1152, 150
            loss_dict, prediction = Postprocess3(out_event, out_event_cls, out_event_duration, is_ischemia.cuda(), event_mask.cuda(), event_class.cuda(), event_duration.cuda(), id)

            weight_dict = {'loss_event_dice': 1, 'loss_event_mse': 10, 'loss_event_cls': 1, 'loss_duration': 1}
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            for i in range(len(prediction)):
                pred_event_cls.append(prediction[i]['event_cls_pred'])
                gt_event_cls.append(prediction[i]['event_cls_gt'])
                gt_event_duration.append(prediction[i]['duration_gt'])

            # accuracy
            acc_event_cls = eval(pred_event_cls, gt_event_cls)
            print('validation loss:  %d, acc_event_cls: %f, loss: %f,  loss_event_dice: %f, loss_event_mse: %f, loss_event_cls: %f, loss_event_duration: %f'
                  % (cnt, acc_event_cls, losses.item(), loss_dict['loss_event_dice'].item(), loss_dict['loss_event_mse'].item(),
                     loss_dict['loss_event_cls'].item(), loss_dict['loss_duration'].item()))
            losses_test = losses.detach().cpu().numpy() + losses_test

        losses_test /= cnt
    return acc_event_cls

def eval(pred_event_cls, gt_event_cls):
    nS = len(pred_event_cls)
    pred_event_cls = torch.tensor(pred_event_cls)
    pred_event_cls[pred_event_cls>0.5] = 1
    pred_event_cls[pred_event_cls <= 0.5] = 0
    gt_event_cls = torch.tensor(gt_event_cls)
    acc_event_cls = pred_event_cls.eq(gt_event_cls).sum().float()/nS/4

    return acc_event_cls

if __name__ == '__main__':
    # Training settings
    use_cuda = torch.cuda.is_available()
    eps = 1e-9
    config = Config()
    batch_size = config.batch_size

    if not os.path.exists(config.backupDir):
        os.mkdir(config.backupDir)

    kwargs = {'num_workers': config.num_workers, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda:%s" % str(config.gpus[0]) if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(config.gpus[0])
        print("GPU is available!")
    else:
        print("GPU is not available!!!")

    model = mcg_bert(input_channel=1,  num_classes=config.num_classes, out_embedding_length=600, hidden=384, n_layers=3)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    writer = None
    ################################
    #         optimizer            #
    ################################
    if config.solver == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate/batch_size, momentum=config.momentum,
                              weight_decay=config.decay*batch_size, nesterov=True)
    elif config.solver == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, betas=config.betas, weight_decay=config.decay, amsgrad=True)
    else:
        print('No %s solver! Please check your config file!' % (config.solver))
        exit()
    ################################
    #         load weights         #
    ################################
    if config.weightFile == None:
        start_epoch = 0
        model = nn.DataParallel(model, device_ids=config.gpus)
        model = model.cuda()
    else:
        checkpoint = torch.load(config.weightFile, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        model = nn.DataParallel(model, device_ids=config.gpus)
        model = model.cuda()

        print('done load model')
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            print('done load optimizer')
        else:
            start_epoch = 0

    train_loader = DataLoader(load_mcg_train(config.imgDirPath, config.labelDirPath, num_classes=config.num_classes),batch_size=config.batch_size, shuffle=True, **kwargs)
    valid_loader = DataLoader(load_mcg_test(config.imgDirPath, config.labelDirPath, num_classes=config.num_classes),batch_size=config.batch_size, shuffle=False, **kwargs)

    for epoch in range(start_epoch, config.max_epochs):

        train_epoch(epoch, train_loader,  config, device, writer)
        loss_valid = evaluate_epoch(valid_loader, model)

        if epoch%config.save_interval == 0:
            torch.save({'epoch': epoch + 1,
                        'optimizer': optimizer.state_dict(),
                        'state_dict': model.module.state_dict()},
                       '{}/checkpoint_layer3_fold2.pth'.format(config.backupDir, 0))

    print('Done!')

