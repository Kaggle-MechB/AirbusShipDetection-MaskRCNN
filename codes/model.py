"""
Mask R-CNN
The main Mask R-CNN model implementation..

Copyright (c) 20017 Matterport, Inc ..
Licensed under the MIT License
Written by Waleed Abdulla
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

import utils
visualize
from nms.nms_wrapper import nms
# cudaのためのツールが入ってる
from roialign.roi_align.crop_and_resize import CropAndResizeFunction
# RoI Align

############################
# ログをとるためのところ
############################

def log(text, array=None):
    if array is not None:
        text = text.ljust(25)
        text +=


##########################
#  Pytorch Utility Functions
##########################

def unique1d(tensor): # ユニークな値一覧を返す
    if tensor.size()[0] == 0 or tensor.size()[0] == 1:
        return tensor
    tensor = tensor.sort()[0] # ソートされた部分だけ返す
    unique_bool = tensor[1:] != tensor[:-1] # 前後にズラしたときの要素が一致するかどうか
    first_element = Variable(torch.ByteTensor([True]), requires_grad=False)
    #
    if tensor.is_cuda:
        first_element = first_element.cuda()
    unique_bool = torch.cat((first_element, unique_bool), dim=0)
    return tensor[unique_bool.data]

def intersect1d(tensor1, tensor2): # 2つのTensorの共通要素を抽出している
    aux = torch.cat((tensor1, tensor2), dim=0)
    aux = aux.sort()[0]
    return aux[:-1][(aux[1:] == aux[:-1]).data]


def log2(x):
    """Implementation of log2 Pytorch doesn't have a native implementation.
    """
    ln2 = Variable(torch.log(torch.FloatTensor([2.0])), requires_grad=False)
    if x.is_cuda:
        ln2 = ln2.cuda()
    return torch.log(x) / ln2

class SamePad2d(nn.Module):
    """Mimics tensorflow's 'SAME' padding..
    """

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.strides = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height)/float(self.stride[1]))
        pad_along_width = ((out_width-1) * self.stride[0] + self.kernel_size[0] - in_width)
        pad_along_height = ((out_heght-1) * self.stride[1] + self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant',value=0)

    def __repr__(self):
        return self.__class__.__name__

############################
# FPN Graph
###########################

class TopDownLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TopDownLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.padding2 = SamePad2d(kernel_size=3,stride=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,stride=1)

    def forward(self, x, y): #
        y = F.upsample(y, scale_factor=2) # 画像サイズを2倍に拡大(単純に引き延ばしてるだけ)
        x = self.conv1(x)
        return self.conv2(self.padding2(x+y))

class FPN(nn.Module):
    def __init__(self, C1, C2, C3, C4, C5, out_channels):
        super(FPN, self).__init__()
        self.out_channels = out_channels
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.C4 = C4
        self.C5 = C5
        self.P6 = nn.MaxPool2d(kernel_size=1, stride=2)
        self.P5_conv1 =  nn.Conv2d(2048, self.out_channels, kernel_size=1, stride=1)
        self.P5_conv2 = nn.Sequential(
           SamePad2d(kernel_size=3, stride=1),
           nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3,stride=1)
        )
        self.P4_conv1 = nn.Conv2d(1024, self.out_channels, kernel_size=1,stride=1)
        self.P4_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3,stride=1),
            nn.Conv2d(self.out_channels,self.out_channels,kernel_size=3,stride=1)
        )
        self.P3_conv1 = nn.Conv2d(512, self.out_channels, kernel_size=1, stride=1)
        self.P3_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels,self.out_channels,kernel_size=3,stride=1)
        )
        self.P2_conv1 = nn.Conv2d(256, self.out_channels, kernel_size=1, stride=1)
        self.P2_conv2 = nn.Sequential(
           SamePad2d(kernel_size=3,stride=1),
           nn.Conv2d(self.out_channels, self.out_channels,kernel_size=3,stride=1)
        )

    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        c2_out = x
        x = self.C3(x)
        c3_out = x
        x = self.C4(x)
        c4_out = x
        x = self.C5(x)
        p5_out = self.P5_conv1(x)
        p4_out = self.P4_conv1(c4_out) + F.upsample(p5_out, scale_factor=2)
        p3_out = self.p3_conv1(c3_out) + F.upsample(p4_out, scale_factor=2)
        p2_out = self.p2_conv1(c2_out) + F.upsample(p3_out, scale_factor=2)

        p5_out = self.P5_conv2(p5_out)
        p4_out = self.P4_conv2(p4_out)
        p3_out = self.P3_conv2(p3_out)
        p2_out = self.P2_conv2(p2_out)

        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2..
        p6_out = self.P6(p5_out)

        return [p2_out, p3_out, p4_out, p5_out, p6_out]

#######################
# ResNet graph
######################

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes, eps=0.001,momentum=0.01)
        self.padding2 = SamePad2d(kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(planes,eps=0.001, momentum=0.01)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(planes*4,eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride= stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.padding2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, architecture, stage5=False):
        super(ResNet, self).__init__()
        # とりあえず50でよさそう
        assert architecture in ["resnet50", "resnet101"]
        self.inplanes = 64
        self.layers = [ 3, 4, {"resnet50":6, "resnet101": 23}[architecture], 3]
        self.block = Bottleneck
        self.stage5 = stage5

        self.c1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            SamePad2d(kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.C2 = self.make_layer(self.block, 64, self.layers[0])
        self.C3 = self.make_layer(self.block, 128, self.layers[1], stride=2)
        self.C4 = self.make_layer(self.block, 256, self.layers[2], stride=2)
        if self.stage5:
            self.C5 = self.make_layer(self.block, 512, self.layers[3], stride=2)
        else:
            self.C5 = None

    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        x = self.C3(x)
        x = self.C4(x)
        x = self.C5(x)
        return x

    def stages(self):
        return [self.C1, self.C2, self.C3, self.C4, self.C5]

    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
              nn.Conv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride),
              nn.BatchNorm2d(planes * block.expansion, eps=0.001, momoentum=0.01),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.implanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

###############
# Proposal Layer
##############

def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes..
    boxes: [N, 4] where each row is y1, x1, y2, x2
    deltas:[N,4] where each row is [dy, dx, leg(dh), log(dw)]
    与えられたデルタの値を元にバウンディングボックスを修正する。
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= torch.exp(deltas[:, 2])
    width *= torch.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = torch.stack([y1, x1, y2, x2], dim=1)
    return result

def clip_boxes(boxes, window):
    """
    boxes: [N, 4] each col is y1, x1, y2, x2
    window:[4] in the form y1, x1, y2, x2
    """
    # tensor.clamp(min, max) で最小最大の範囲をはみ出してたら最小か最大の値を返す
    # 今回はx1,x2,y1,y2
    boxes = torch.stack(\
        [boxes[:, 0].clamp(float(window[0]), float(window[2])),
         boxes[:, 1].clamp(float(window[1]), float(window[3])),
         boxes[:, 2].clamp(float(window[0]), float(window[2])),
         boxes[:, 3].clamp(float(window[1]), float(window[3]))], dim=1)
    return boxes

# anchors: torchのTensor。[anchors, (y1, x1, y2, x2)]。領域候補の集合。
def proposal_layer(inputs, proposal_count, nms_threshold, anchors, config=None):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps.. It also applies bounding
    box refinment details to anchors. .

    Inputs:
        rpn_probs: [batch, anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
        anchorsはanchorの数

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    # Currently only supports batchsize 1
    inputs[0] = inputs[0].squeeze(0)
    inputs[1] = inputs[1].squeeze(0)
    # dim=1の次元数が1のときその次元を取り除く

    # Box Scores foregrouncクラスの根拠に使う [Batch, num_rois, 1]
    scores = inputs[0][:, 1] # fg probを取り出す
    # Box deltas [batch, num_rois, 4]
    deltas = inputs[1] # rpn_bbox
    std_dev = Variable(torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV,
                                [1,4])).float(), requires_grad=False)
    # std = [0,1, 0.1, 0.2, 0.2]

     if config.GPU_COUNT:
         std_dev = std_dev.cuda()
    deltas = deltas * std_dev

    # Improve performance by trimming to top anchors by score
    # and doing the rest on the smaller subset..
    pre_nms_limit = min(6000, anchors.size()[0])
    # fg probsの値が高い順にデータをそれぞれとってくる。
    scores, order = scores.sort(descending=True)
    # ソートされる前の値のインデックス
    order = order[:pre_nms_limit]
    scores =scores[:pre_nms_limit]
    deltas = deltas[order.data, :]  # TODO: Support batch size > 1 ff..
    anchors = anchors[order.data, :]

    boxes = apply_box_deltas(anchors, deltas)

    # Clip to image boundaries.. [y1, x1, y2, x2]
    height, width = config.IMAGE_SHAPE[:2] # 画像本来の大きさ [1024, 1024, 3]ここは変更予定
    window = np.array([0, 0, height, width]).astype(np.float32)
    boxes = clip_boxes(boxes, window)
    # 画像からはみ出てるバウンディングボックスを変形する

    # Filter out small boxes
    # According to Xinlei Chen's paper, this reduces detection accuracy

    # Non-max suppression
    # nms_thresholdを超えた値を削除している
    keep = nms(torch.cat((boxes, scores.unsqueeze(1)), 1).data, nms_threshold)
    # keepはインデックス?
    keep = keep[:proposal_count]
    boxes = boxes[keep, :]

    # Normalize dimensions to range of 0 to 1. .
    # 相対的な値に変換する
    norm = Variable(torch.from_numpy(np.array([height, widht, height, width]))\
                                         .float(), requires_grad=False)
    if config.GPU_COUNT:
        norm = norm.cuda()
    normalized_boxes = boxes / norm

    normalize_boxed = normalized_boxes.unsqueeze(0)
    # 元に戻す

    return normalized_boxes

#########################
# ROIAlign Layer
#########################

def pyramid_roi_align(inputs, pool_size, image_shape):
    """Implements ROI Pooling on multiple levels of the feature pyramid..

    Params:
    - pool_size: [height, width] of the output pooled regions Usually [7,7]
    - image_shape: [height, width, channels] Shape of input image in pixels

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates..
    - Features maps: List of feature maps from different levels of the pyramid..
                     Each is [batch, channels, height, width]

    Output:
    Pooled regions in the shape: [num_boxes, height, width, channels]..
    The width and height are those specific in hte pool_shape in the layer
    constructor .
    """

    # Currently only supports batchsize 1
    for i in range(len(inputs)):
        inputs[i] = inputs[i].squeeze(0)

    # Crops boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
    boxes = inputs[0]

    # Feature Maps.. List of feature maps from different level of the
    # feature pyramid.. Each is [batch, height, width, channels]
    features_maps = inputs[1:]

    # Assign each ROI to a level in the pyramid base on the ROI area
    y1, x1, y2, x2 = boxes.chunk(4, dim=1) # バラしてる
    h = y2 - y1
    w = x2 - x1

    # Equation 1 in the Feature Pyramid Networks paper Account for
    # the fact that our coordinates are normalized here..
    # e.g.. a 224x224 ROI (in pixels) maps to P4
    image_area = Variable(torch.FloatTensor([float(image_shape[0]*image_shape[1])]),
                    requires_grad=False)
    if boxes.is_cuda:
        image_area = image_area.cuda()
    roi_level = 4 + log2(torch.sqrt(h*w)/224.0/torch.sqrt(image_area))
    roi_level = roi_level.round().int()
    roi_level = roi_level.clamp(2,5)
    # 抽出領域と全体の画像との大きさの比率でP2~P5のどの特徴量を使うかを決めている
    # Loop through levels and apply ROI pooling to each P2 to P5
    pooled = []
    box_to_level = []
    for i, level in enumerate(range(2,6)):
        ix = roi_level == level
        if not ix.any():
            continue
        ix = torch.nonzero(ix)[:, 0]
        level_boxes = boxes[ix.data, :]

        # keep track of which obx iis mapped to which level
        box_to_level.append(ix.data)

        # Stop gradient propagation to ROI proposals
        level_boxes = level_boxes.detach()

        # Crop and Resize
        # From Mask R-CNN paper: "We sample four regular locations, so
        # that we can evaluate either max or average pooling. IN Fact,
        # interpolating only a single value at each bin center(without
        # pooling) is nearly as effective."
        #
        # Here we use the simplified approach of a single value per bin,
        # which is how it's done in tf.crop_and_resize()
        # Result: [batch * num_boxes, pool_height, pool_width, channels]
        ind = Variable(torch.zeros(level_boxes.size()[0]), requires_grad=False).int()
        if level_boxes.is_cuda:
            ind = ind.cuda()
        features_maps[i] = features_maps[i].unsqueeze(0)
        pooled_features = CropAndResizeFunction(pool_size, pool_size, 0)(feature_maps[i], level_boxes, ind)
        pooled.append(pooled_features)

    # Pack pooled features into one tensor
    pooled = troch.cat(pooled, dim=0)

    # Pack box_to_level mapping into one array and add another
    # column representing the order of pooled boxes
    box_to_level = torch.cat(box_to_level, dim=0)

    _, box_to_level = torch.sort(box_to_level)
    pooled = pooled[box_to_level, :, :]

    return pooled
