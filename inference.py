# encoding: utf-8
from __future__ import print_function
import torch.optim as optim
from torch.utils.data import DataLoader
from models import mcg_bert, Postprocess3
import time, logging, os
from config import Config
from utils import *
from dataset import load_mcg_train, load_mcg_test
import torch
import torch.nn as nn
import pickle
import wandb
# wandb.init(project="GANsyn")
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

def evaluate_epoch(valid_loader, model):
    model.eval()
    print('-----------Start Inference------------')
    with torch.no_grad():
        losses_test = 0
        cnt = 0
        pred_event_duration = []
        gt_event_duration = []
        pred_event_cls=[]
        gt_event_cls=[]
        for batch_idx, (amcg, is_ischemia, event_mask, event_class, event_duration, id) in enumerate(valid_loader):
            cnt += amcg.shape[0]
            out_event, out_event_cls, out_event_duration, attn = model(amcg.cuda(), mode='test')
            loss_dict, prediction = Postprocess3(out_event, out_event_cls, out_event_duration, is_ischemia.cuda(), event_mask.cuda(), event_class.cuda(), event_duration.cuda(), id)

            # save attn
            out_path = '/media/gdp/date/MCG_project/MCG/pred/attn'
            for i in range(len(id)):
                a = attn[0,:,:,:].detach().cpu().numpy()
                data = {'attn':a, 'prediction':prediction[i]}
                with open(os.path.join(out_path, id[i]+ '.pickle'), 'wb') as f:
                    pickle.dump(data, f)
                f.close()

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
            # break
        losses_test /= cnt

    return losses_test

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

    model = mcg_bert(input_channel=1, num_classes=config.num_classes, out_embedding_length=600, hidden=384, n_layers=3) # best at 31
    #
    checkpoint = torch.load('backup/checkpoint_layer6_bestacc.pth', map_location='cpu'),
    print('epoch:',checkpoint[0]['epoch'] )


    model.load_state_dict(checkpoint[0]['state_dict'])
    print('done load model')

    config.gpus = [0]
    model = nn.DataParallel(model, device_ids=config.gpus)
    model = model.cuda()
    valid_loader = DataLoader(load_mcg_test(config.imgDirPath, config.labelDirPath, num_classes=config.num_classes, fold='1'),batch_size=30, shuffle=False, **kwargs)
    loss_valid_new = evaluate_epoch(valid_loader, model)

