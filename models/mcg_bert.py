import torch
import torch.nn as nn
from models.Models import NLWapperEncoder
import torch.nn.functional as F
#
class BottleNeck1d(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, size, stride=[2,1], downsample=None, conv_num=3):
        super(BottleNeck1d, self).__init__()
        self.same_padding = size//2
        self.layers1 = []
        for i in range(conv_num):
            if i == 0:
                self.layers1.append(nn.Conv1d(inplanes, planes, kernel_size=size, stride=stride[0], padding=self.same_padding, bias=False))
                self.layers1.append(nn.BatchNorm1d(planes))
                self.layers1.append(nn.ReLU(inplace=True))
            else:
                self.layers1.append(nn.Conv1d(planes, planes, kernel_size=size, stride=1, padding=self.same_padding, bias=False))
                self.layers1.append(nn.BatchNorm1d(planes))
                self.layers1.append(nn.ReLU(inplace=True))
        self.layers1 = nn.Sequential(*self.layers1)

        self.layers2 = []
        for i in range(conv_num):
            if i == 0:
                self.layers2.append(nn.Conv1d(planes, planes, kernel_size=size, stride=stride[1], padding=self.same_padding, bias=False))
                self.layers2.append(nn.BatchNorm1d(planes))
                self.layers2.append(nn.ReLU(inplace=True))
            else:
                self.layers2.append(nn.Conv1d(planes, planes, kernel_size=size, stride=1, padding=self.same_padding, bias=False))
                self.layers2.append(nn.BatchNorm1d(planes))
                self.layers2.append(nn.ReLU(inplace=True))
        self.layers2 = nn.Sequential(*self.layers2)

        self.neck = []
        self.neck.append(nn.Conv1d(inplanes, planes, kernel_size=size, stride=stride[0], padding=self.same_padding, bias=False))
        self.neck.append(nn.BatchNorm1d(planes))
        self.neck.append(nn.ReLU(inplace=True))
        self.neck = nn.Sequential(*self.neck)

        self.seblock = SEblock1D(planes)
        self.downsample = downsample

    def forward(self, x):
        neck = self.neck(x)

        out = self.layers1(x)
        out = self.seblock(out)

        out += neck

        residual = out
        out = self.layers2(out)
        out = self.seblock(out)
        out += residual

        return out

class BasicBlock3d(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, size, stride=1, downsample=None, conv_num=3):
        super(BasicBlock3d, self).__init__()
        self.layers = []
        self.same_padding = size//2
        for i in range(conv_num):
            if i == 0:
                self.layers.append(nn.Conv3d(inplanes, planes, kernel_size=(3,3,size), stride=stride, padding=(1,1,self.same_padding), bias=False))
                self.layers.append(nn.BatchNorm3d(planes))
                self.layers.append(nn.ReLU(inplace=True))
            else:
                self.layers.append(nn.Conv3d(planes, planes, kernel_size=(3,3,size), stride=(1,1,1), padding=(1,1,self.same_padding), bias=False))
                self.layers.append(nn.BatchNorm3d(planes))
                self.layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*self.layers)
        self.seblock = SEblock3D(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.layers(x)
        out = self.seblock(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

class SEblock3D(nn.Module):
    def __init__(self, inplanes):
        super(SEblock3D, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool3d(1)
        self.conv1 = nn.Conv3d(inplanes, int(inplanes / 16), kernel_size=1, stride=1)
        self.conv2 = nn.Conv3d(int(inplanes / 16), inplanes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out = self.global_avgpool(x)

        out = self.conv1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.sigmoid(out)

        return x * out

class SEblock2D(nn.Module):
    def __init__(self, inplanes):
        super(SEblock2D, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(inplanes, int(inplanes / 16), kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(int(inplanes / 16), inplanes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out = self.global_avgpool(x)

        out = self.conv1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.sigmoid(out)

        return x * out

class SEblock1D(nn.Module):
    def __init__(self, inplanes):
        super(SEblock1D, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool1d(1)
        self.conv1 = nn.Conv1d(inplanes, int(inplanes / 16), kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(int(inplanes / 16), inplanes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.sigmoid(out)
        return x * out

class backbone(nn.Module):
    def __init__(self, input_channel=1, hidden=384, out_embedding_length=300):
        super(backbone, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(input_channel, 32, kernel_size=(1,1,15), stride=(1,1,1), padding=(0, 0, 7), bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.ks = [15,5,7,15]
        # pooling can be removed
        self.avgpool = nn.AdaptiveAvgPool1d(out_embedding_length)

        # block1
        self.inplanes = 32
        self.layers1 = nn.Sequential()
        self.layers1.add_module('layer_1_1', self._make_layer3d(BasicBlock3d, self.inplanes, 1, stride=(1, 1, 1), size=self.ks[0], conv_num=1))

        # block2
        self.layers2_1 = nn.Sequential()
        self.layers2_1.add_module('layer_2_1_1', self._make_layer3d(BasicBlock3d, self.inplanes, 1, stride=(1, 1, 1), size=self.ks[1], conv_num=2))
        self.layers2_1.add_module('layer_2_1_4', self._make_layer3d(BasicBlock3d, self.inplanes, 1, stride=(1, 1, 2), size=self.ks[1], conv_num=2))

        self.layers2_2 = nn.Sequential()
        self.layers2_2.add_module('layer_2_2_1', self._make_layer3d(BasicBlock3d, self.inplanes, 1, stride=(1, 1, 1), size=self.ks[2], conv_num=2))
        self.layers2_2.add_module('layer_2_2_4', self._make_layer3d(BasicBlock3d, self.inplanes, 1, stride=(1, 1, 2), size=self.ks[2], conv_num=2))

        self.layers2_3 = nn.Sequential()
        self.layers2_3.add_module('layer_2_3_1', self._make_layer3d(BasicBlock3d, self.inplanes, 1, stride=(1, 1, 1), size=self.ks[3], conv_num=2))
        self.layers2_3.add_module('layer_2_3_4', self._make_layer3d(BasicBlock3d, self.inplanes, 1, stride=(1, 1, 2), size=self.ks[3], conv_num=2))

        # block3
        self.inplanes *= 36
        self.layers3_1 = nn.Sequential()
        self.layers3_1.add_module('layer_3_1_1', self._make_layer1d(BottleNeck1d, self.inplanes, 1, stride=[1, 1], size=self.ks[1], conv_num=3))
        self.layers3_1.add_module('layer_3_1_4', self._make_layer1d(BottleNeck1d, self.inplanes, 1, stride=[2, 1], size=self.ks[1], conv_num=3))

        self.layers3_2 = nn.Sequential()
        self.layers3_2.add_module('layer_3_2_1', self._make_layer1d(BottleNeck1d, self.inplanes, 1, stride=[1, 1], size=self.ks[2], conv_num=3))
        self.layers3_2.add_module('layer_3_2_4', self._make_layer1d(BottleNeck1d, self.inplanes, 1, stride=[2, 1], size=self.ks[2], conv_num=3))

        self.layers3_3 = nn.Sequential()
        self.layers3_3.add_module('layer_3_3_1', self._make_layer1d(BottleNeck1d, self.inplanes, 1, stride=[1, 1], size=self.ks[3], conv_num=3))
        self.layers3_3.add_module('layer_3_3_4', self._make_layer1d(BottleNeck1d, self.inplanes, 1, stride=[2, 1], size=self.ks[3], conv_num=3))

        self.fc = nn.Linear(self.inplanes * (len(self.ks)-1), hidden)

    def _make_layer3d(self, block, planes, blocks, stride=(1, 1, 2), size=15, conv_num=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, size, stride, downsample, conv_num))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_layer1d(self, block, planes, blocks, stride=2, size=15, conv_num=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, size, stride, downsample, conv_num))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x0):
        x0 = self.relu(self.bn1(self.conv1(x0)))
        #block1
        x0 = self.layers1(x0)

        #block2
        x2_1 = self.layers2_1(x0)
        x2_1 = torch.flatten(x2_1, start_dim=1, end_dim=3)
        x2_2 = self.layers2_2(x0)
        x2_2 = torch.flatten(x2_2, start_dim=1, end_dim=3)
        x2_3 = self.layers2_3(x0)
        x2_3 = torch.flatten(x2_3, start_dim=1, end_dim=3)

        #block3
        x3_1 = self.layers3_1(x2_1)
        # x3_1 = self.avgpool(x3_1)
        x3_2 = self.layers3_1(x2_2)
        # x3_2 = self.avgpool(x3_2)
        x3_3 = self.layers3_1(x2_3)
        # x3_3 = self.avgpool(x3_3)

        out = torch.cat([x3_1, x3_2, x3_3], dim=1).permute(0,2,1)
        out = self.fc(out)
        return out

class mcg_bert(nn.Module):
    def __init__(self, input_channel=1,  num_classes=5, out_embedding_length=279, hidden=384, n_layers=3):
        super(mcg_bert, self).__init__()
        self.hidden = hidden
        self.seq_len = out_embedding_length
        self.backbone = backbone(input_channel, hidden, out_embedding_length)
        self.encoder = NLWapperEncoder(n_layers=n_layers, hidden=hidden)  # 1, 384, 300
        self.classifier_event = nn.Linear(hidden, num_classes)
        self.classifier_event_cls = nn.Linear(hidden, 1)
        self.event_duration = nn.Linear(hidden, 1)

    def forward(self, x0, mode='train'):
        embedding = self.backbone(x0)  # 1, 384, 300
        if mode == 'train':
            embedding = F.interpolate(embedding.permute(0, 2, 1), size=(self.seq_len), mode='linear').permute(0,2,1)
        else:
            embedding = F.interpolate(embedding.permute(0, 2, 1), size=(x0.shape[-1]), mode='linear').permute(0, 2, 1)
        out, attn = self.encoder(embedding)
        out = out + embedding
        out_event = self.classifier_event(out)
        out_event_cls = self.classifier_event_cls(out).squeeze(-1)
        out_event_duration = self.event_duration(out).squeeze(-1)
        return out_event, out_event_cls, out_event_duration, attn[-1]

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def dice_loss(inputs, targets):
    inputs = inputs.flatten(0)
    targets = targets.flatten(0)
    numerator = 2*torch.dot(inputs, targets).sum(-1)
    denominator = torch.dot(inputs, inputs).sum(-1) + torch.dot(targets, targets).sum(-1)
    loss = 1 - (numerator + 1e-9) / (denominator + 1e-9)
    return loss

def Criterion3(out_event, out_event_cls, out_event_duration, is_ischemia, event_mask, event_class, event_duration, use_dice=True):
    # out_event = F.interpolate(out_event.permute(0,2,1),size=(event_mask.shape[-1]), mode='linear').permute(0,2,1)
    # out_event_cls = F.interpolate(out_event_cls[:,None,:],size=(event_mask.shape[-1]), mode='linear').squeeze(1)
    # event detection
    event_mask = event_mask.permute(0,2,1)
    loss_event_mse = nn.MSELoss()(out_event, event_mask)
    if use_dice:
        # we find that use a modified version of dice loss (with relu activation) can speedup network convergence
        loss_event_dice = dice_loss(torch.relu(out_event), event_mask)

    # event_classification # index 0--> background
    pred_event_cls = torch.zeros(event_class.shape).to(out_event_cls.device)
    pred_event_duration = torch.zeros(event_class.shape).to(out_event_cls.device)
    store_ind = torch.zeros((pred_event_cls.shape[0], 4))
    for i in range(4):
        ind = torch.argmax(event_mask[:,:,i+1], dim=1)
        store_ind[:,i] = ind
        for j in range(len(ind)):
            pred_event_cls[j, i] = out_event_cls[j, ind[j]]
            pred_event_duration[j, i] = out_event_duration[j, ind[j]]


    loss_event_cls = nn.BCELoss()(torch.sigmoid(pred_event_cls), event_class)
    loss_event_duration = nn.L1Loss()(torch.sigmoid(pred_event_duration), event_duration)

    losses = {'loss_duration':loss_event_duration, 'loss_event_dice':loss_event_dice,'loss_event_mse':loss_event_mse,'loss_event_cls':loss_event_cls}
    # print(losses)

    predictions = []
    store_ind = store_ind.detach().cpu().numpy()
    for j in range(is_ischemia.shape[0]):
        predictions.append({'event_cls_pred':list(torch.sigmoid(pred_event_cls[j,:]).detach().cpu().numpy()),
                            'Q_pred':store_ind[j,0],'R_pred':store_ind[j,1],'S_pred':store_ind[j,2],'T_pred':store_ind[j,3],
                            'duration_pred':list(torch.sigmoid(pred_event_duration[j,:]).detach().cpu().numpy()),
                           'event_cls_gt':list(event_class[j,:].detach().cpu().numpy()),'duration_gt':event_duration[j].detach().cpu().numpy()})
        print(predictions[-1])

    return losses

def Postprocess3(out_event, out_event_cls, out_event_duration, is_ischemia, event_mask, event_class, event_duration, id=None):
    # out_event = F.interpolate(out_event.permute(0,2,1),size=(event_mask.shape[-1]), mode='linear').permute(0,2,1)
    # out_event_cls = F.interpolate(out_event_cls[:,None,:],size=(event_mask.shape[-1]), mode='linear').squeeze(1)
    # event detection
    event_mask = event_mask.permute(0,2,1)
    print(out_event.shape, event_mask.shape)
    loss_event_dice = dice_loss(torch.relu(out_event), event_mask)
    loss_event_mse = nn.MSELoss()(out_event, event_mask)

    # event_classification # index 0--> background
    pred_event_cls = torch.zeros(event_class.shape).to(out_event_cls.device)
    pred_event_duration = torch.zeros(event_class.shape).to(out_event_cls.device)
    pred_event_mask = torch.relu(out_event)
    store_ind = torch.zeros((pred_event_cls.shape[0], 4))
    for i in range(pred_event_mask.shape[0]):
        for j in range(4):
            if torch.max(pred_event_mask[i,:,j+1]) <= 0.1:
                print(id, 'no event', j, 'detected')
            else:
                ind = torch.argmax(pred_event_mask[i, :, j + 1])
                store_ind[i, j] = ind
                pred_event_cls[i, j] = out_event_cls[i, ind]
                pred_event_duration[i, j] = out_event_duration[i, ind]

    # gt ind
    store_ind_gt = torch.zeros((pred_event_cls.shape[0], 4))
    for i in range(4):
        ind = torch.argmax(event_mask[:,:,i+1], dim=1)
        store_ind_gt[:,i] = ind

    loss_event_cls = nn.BCELoss()(torch.sigmoid(pred_event_cls), event_class)
    loss_event_duration = nn.L1Loss()(torch.sigmoid(pred_event_duration), event_duration)
    losses = {'loss_duration':loss_event_duration, 'loss_event_dice':loss_event_dice, 'loss_event_mse':loss_event_mse, 'loss_event_cls':loss_event_cls}

    predictions = []
    store_ind = store_ind.detach().cpu().numpy()
    for j in range(is_ischemia.shape[0]):
        predictions.append({'id':id[j],'event_cls_pred':list(torch.sigmoid(pred_event_cls[j,:]).detach().cpu().numpy()),
                            'Q_pred':store_ind[j,0],'R_pred':store_ind[j,1],'S_pred':store_ind[j,2],'T_pred':store_ind[j,3],
                            'duration_pred': list(torch.sigmoid(pred_event_duration[j,:]).detach().cpu().numpy()),
                            'event_cls_gt':list(event_class[j,:].detach().cpu().numpy()),'ischemia_gt':is_ischemia[j].detach().cpu().numpy(),
                            'Q_gt':store_ind_gt[j,0],'R_gt':store_ind_gt[j,1],'S_gt':store_ind_gt[j,2],'T_gt':store_ind_gt[j,3],
                            'duration_gt':event_duration[j].detach().cpu().numpy()})
        print(predictions[-1])
    return losses, predictions



















