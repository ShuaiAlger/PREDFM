#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 13:46:43 2021

@author: kutalmisince
"""
import numpy as np
import cv2 as cv
import torch
from torchvision import models, transforms
from collections import namedtuple
import matplotlib.pyplot as plt
from torchvision.models.feature_extraction import create_feature_extractor


from torchvision.models.efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3
from torchvision.models.efficientnet import efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
from torchvision.models.efficientnet import efficientnet_v2_l, efficientnet_v2_m, efficientnet_v2_s
# from torchvision.models.maxvit import maxvit_t
from torchvision.models.mnasnet import mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3
from torchvision.models.regnet import regnet_x_8gf, regnet_x_16gf, regnet_x_1_6gf, regnet_x_32gf, regnet_x_3_2gf, regnet_x_400mf, regnet_x_800mf

from torchvision.models.alexnet import alexnet
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, wide_resnet50_2, wide_resnet101_2
from torchvision.models.densenet import densenet121, densenet161, densenet169, densenet201

from torchvision.models.resnet import resnext50_32x4d, resnext101_64x4d, resnext101_32x8d
from torchvision.models.squeezenet import squeezenet1_0, squeezenet1_1
from torchvision.models.mobilenet import mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small

from torchvision.models.googlenet import googlenet
from torchvision.models.inception import inception_v3

from torchvision.models.vgg import vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from torchvision.models.shufflenetv2 import shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0
from torchvision.models.convnext import convnext_base, convnext_large, convnext_small, convnext_tiny


class ParamMatcher(torch.nn.Module):
    def __init__(self, model: str = 'resnet18', device = None,
                 bidirectional=True,
                 enable_two_stage = True,
                 ratio_th = [0.9, 0.9, 0.9, 0.9, 0.95, 1.0],
                 layers = []):
        super(ParamMatcher, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device == None else device        
        
        self.enable_two_stage = enable_two_stage
        self.bidirectional = bidirectional
        self.ratio_th = np.array(ratio_th)

        self.model_name = model

        self.combination_index = 0

        if model == 'resnet18':
            self.stage1 = [
                'relu'
            ]
            self.stage2 = (
                'maxpool',
                'layer1.0.relu',
                'layer1.0',
                'layer1.1.relu',
                'layer1.1'
            )
            self.stage3 = [
                'layer2.0.relu',
                'layer2.0',
                'layer2.1.relu',
                'layer2.1'
            ]
            self.stage4 = [
                'layer3.0.relu',
                'layer3.0',
                'layer3.1.relu',
                'layer3.1'
            ]

            self.return_nodes_list = []

            for i1 in range(len(self.stage1)):
                for i2 in range(len(self.stage2)):
                    for i3 in range(len(self.stage3)):
                        for i4 in range(len(self.stage4)):
                            tmp_return_nodes = {
                                self.stage1[i1]: self.stage1[i1],
                                self.stage2[i2]: self.stage2[i2],
                                self.stage3[i3]: self.stage3[i3],
                                self.stage4[i4]: self.stage4[i4],
                            }
                            self.return_nodes_list.append(tmp_return_nodes)

            i1 = i2 = i3 = i4 = 0
            self.return_nodes = {
                self.stage1[i1]: self.stage1[i1],
                self.stage2[i2]: self.stage2[i2],
                self.stage3[i3]: self.stage3[i3],
                self.stage4[i4]: self.stage4[i4],
            }

            self.combination_nums = len(self.return_nodes_list)

            self.smodel = resnet18(pretrained=True).to(self.device)

            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes_list[self.combination_index]).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'resnet34':
            self.return_nodes = {
                'relu': 'relu',
                'layer1': 'layer1',
                'layer2': 'layer2',
                'layer3': 'layer3',
            }
            self.smodel = resnet34(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'resnet50':
            self.return_nodes = {
                'relu': 'relu',
                'layer1': 'layer1',
                'layer2': 'layer2',
                'layer3': 'layer3',
            }
            self.smodel = resnet50(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        else:
            print('Error: model ' + model + ' is not supported!')
            return
        
        self.enable_two_stage = enable_two_stage
        self.bidirectional = bidirectional
        self.ratio_th = np.array(ratio_th)

    def get_combination_nums(self):
        return self.combination_nums

    def get_combination_names(self, index):
        if index < self.combination_nums:
            return self.return_nodes_list[index]
        else:
            return []

    def next_combination(self):
        if self.combination_index < len(self.return_nodes_list):
            self.combination_index = self.combination_index + 1
            self.return_nodes = self.return_nodes_list[self.combination_index]
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            # print('model is loaded : ', self.return_nodes)
        else:
            print("============>          Combination is END            <==============")

    def set_combination(self, index):
        if index < len(self.return_nodes_list):
            self.return_nodes = self.return_nodes_list[index]
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            # print('model is loaded : ', self.return_nodes)
        else:
            print("============>          Combination index is WRONG            <==============")


    def match(self, img_A, img_B, display_results=0, *args):
        # transform into pytroch tensor and pad image to a multiple of 16
        inp_A, padding_A = self.transform(img_A) 
        inp_B, padding_B = self.transform(img_B)

        outputs_A = self.model(inp_A)
        outputs_B = self.model(inp_B)

        activations_A = []
        activations_B = []

        for x in self.return_nodes:
            activations_A.append(outputs_A[str(x)].to(self.device))
            activations_B.append(outputs_B[str(x)].to(self.device))

        if self.model_name[0:3] == "vgg":
            print(self.model_name[0:3])
        else:
            for i in range(len(activations_A)):
                activations_A[i] = self.upsample(activations_A[i])
                activations_B[i] = self.upsample(activations_B[i])

        # for i in range(len(activations_A)):
        #     print(activations_A[i].shape)

        # initiate warped image, its activations, initial&final estimate of homography
        img_C = img_B
        activations_C = activations_B
        H_init = np.eye(3, dtype=np.double)
        H = np.eye(3, dtype=np.double)

        # initiate matches
        # points_A, points_C = dense_feature_matching(activations_A[-2], activations_C[-2], self.ratio_th[-2], self.bidirectional)

        # for k in range(len(activations_A) - 3, -1, -1):
        #     points_A, points_C = refine_points(points_A, points_C, activations_A[k], activations_C[k], self.ratio_th[k], self.bidirectional)

        points_A, points_C = dense_feature_matching(activations_A[-1], activations_C[-1], self.ratio_th[-1], self.bidirectional)

        for k in range(len(activations_A) - 2, -1, -1):
            points_A, points_C = refine_points(points_A, points_C, activations_A[k], activations_C[k], self.ratio_th[k], self.bidirectional)
            

            if display_results == 2:
                self.plot_keypoints(img_A, (points_A + 0.5) * (2**k) - 0.5, 'A level: ' + str(k))
                self.plot_keypoints(img_C, (points_C + 0.5) * (2**k) - 0.5, 'Bw level: ' + str(k))

        # warp points form C to B (H_init is zero-based, use zero-based points)
        points_B = torch.from_numpy(np.linalg.inv(H_init)) @ torch.vstack((points_C, torch.ones((1, points_C.size(1))))).double()
        points_B = points_B[0:2, :] / points_B[2, :]

        points_A = points_A.double()

        # optional
        in_image = torch.logical_and(points_A[0, :] < (inp_A.shape[3] - padding_A[0] - 16), points_A[1, :] < (inp_A.shape[2] - padding_A[1] - 16))
        in_image = torch.logical_and(in_image, 
                                     torch.logical_and(points_B[0, :] < (inp_B.shape[3] - padding_B[0] - 16), points_B[1, :] < (inp_B.shape[3] - padding_B[1] - 16)))

        points_A = points_A[:, in_image]
        points_B = points_B[:, in_image]

        # estimate homography
        src = points_B.t().numpy()
        dst = points_A.t().numpy()

        if points_A.size(1) >= 4:
            H, _ = cv.findHomography(src, dst, method=cv.RANSAC, ransacReprojThreshold=3.0, maxIters=5000, confidence=0.9999)

        # opencv might return None for H, check for None
        H = np.eye(3, dtype=np.double) if H is None else H

        # return H, H_init, points_A.numpy(), points_B.numpy()
        return points_A.numpy().T, points_B.numpy().T





    def transform(self, img):
        # transform to tensor and normalize
        T = transforms.Compose([transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.to(self.device)),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
                                ])
        
        # zero padding to make image canvas a multiple of padding_n
        pad_right = 16 - img.shape[1] % self.padding_n if img.shape[1] % self.padding_n else 0
        pad_bottom = 16 - img.shape[0] % self.padding_n if img.shape[0] % self.padding_n else 0
        
        padding = torch.nn.ZeroPad2d([0, pad_right, 0, pad_bottom])

        # convert image
        #img_T = padding(T(img.astype(np.float32))).unsqueeze(0)
        img_T = padding(T(img)).unsqueeze(0)

        return img_T, (pad_right, pad_bottom)  
    
    @classmethod  
    def plot_keypoints(cls, img, pts, title='untitled', *args):
    
        f,a = plt.subplots()
        if len(args) > 0:
            pts2 = args[0]
            a.plot(pts2[0, :], pts2[1, :], marker='o', linestyle='none', color='green')
        
        a.plot(pts[0, :], pts[1, :], marker='+', linestyle='none', color='red')
        a.imshow(img)
        a.title.set_text(title)
        plt.pause(0.001)
        #plt.show() 
          



def dense_feature_matching(map_A, map_B, ratio_th, bidirectional=True):
    # normalize and reshape feature maps
    _, ch, h_A, w_A = map_A.size()
    _, _,  h_B, w_B = map_B.size()

    d1_ = map_A.view(ch, -1).t()
    d1 = d1_/torch.sqrt(torch.sum(torch.square(d1_), 1)).unsqueeze(1)

    d2_ = map_B.view(ch, -1).t()
    d2 = d2_/torch.sqrt(torch.sum(torch.square(d2_), 1)).unsqueeze(1)

    # # perform matching
    matches, scores = mnn_ratio_matcher(d1, d2, ratio_th, bidirectional)

    # # form a coordinate grid and convert matching indexes to image coordinates
    y_A, x_A = torch.meshgrid(torch.arange(h_A), torch.arange(w_A))
    y_B, x_B = torch.meshgrid(torch.arange(h_B), torch.arange(w_B))

    points_A = torch.stack((x_A.flatten()[matches[:, 0]], y_A.flatten()[matches[:, 0]]))
    points_B = torch.stack((x_B.flatten()[matches[:, 1]], y_B.flatten()[matches[:, 1]]))

    # discard the point on image boundaries
    discard = (points_A[0, :] == 0) | (points_A[0, :] == w_A-1) | (points_A[1, :] == 0) | (points_A[1, :] == h_A-1) \
            | (points_B[0, :] == 0) | (points_B[0, :] == w_B-1) | (points_B[1, :] == 0) | (points_B[1, :] == h_B-1)

    #discard[:] = False
    points_A = points_A[:, ~discard]
    points_B = points_B[:, ~discard]

    return points_A, points_B





def refine_points(points_A: torch.Tensor, points_B: torch.Tensor, activations_A: torch.Tensor, activations_B: torch.Tensor, ratio_th = 0.9, bidirectional = True):

    # normalize and reshape feature maps
    d1 = activations_A.squeeze(0) / activations_A.squeeze(0).square().sum(0).sqrt().unsqueeze(0)
    d2 = activations_B.squeeze(0) / activations_B.squeeze(0).square().sum(0).sqrt().unsqueeze(0)
        
    # get number of points
    ch = d1.size(0)
    num_input_points = points_A.size(1)
    
    if num_input_points == 0:
        return points_A, points_B
    
    # upsample points
    points_A *= 2
    points_B *= 2
       
    # neighborhood to search
    neighbors = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])

    # allocate space for scores
    scores = torch.zeros(num_input_points, neighbors.size(0), neighbors.size(0))

    # for each point search the refined matches in given [finer] resolution
    for i, n_A in enumerate(neighbors):   
        for j, n_B in enumerate(neighbors):

            # get features in the given neighborhood
            act_A = d1[:, points_A[1, :] + n_A[1], points_A[0, :] + n_A[0]].view(ch, -1)
            act_B = d2[:, points_B[1, :] + n_B[1], points_B[0, :] + n_B[0]].view(ch, -1)

            # compute mse
            scores[:, i, j] = torch.sum(act_A * act_B, 0)
            
    # retrieve top 2 nearest neighbors from A2B
    score_A, match_A = torch.topk(scores, 2, dim=2)
    score_A = 2 - 2 * score_A
    
    # compute lowe's ratio
    ratio_A2B = score_A[:, :, 0] / (score_A[:, :, 1] + 1e-8)
    
    # select the best match
    match_A2B = match_A[:, :, 0]
    score_A2B = score_A[:, :, 0]
    
    # retrieve top 2 nearest neighbors from B2A
    score_B, match_B = torch.topk(scores.transpose(2,1), 2, dim=2)
    score_B = 2 - 2 * score_B
    
    # compute lowe's ratio
    ratio_B2A = score_B[:, :, 0] / (score_B[:, :, 1] + 1e-8)
    
    # select the best match
    match_B2A = match_B[:, :, 0]
    #score_B2A = score_B[:, :, 0]
    
    # check for unique matches and apply ratio test
    ind_A = (torch.arange(num_input_points).unsqueeze(1) * neighbors.size(0) + match_A2B).flatten()
    ind_B = (torch.arange(num_input_points).unsqueeze(1) * neighbors.size(0) + match_B2A).flatten()
    
    ind = torch.arange(num_input_points * neighbors.size(0))
    
    # if not bidirectional, do not use ratios from B to A
    ratio_B2A[:] *= 1 if bidirectional else 0 # discard ratio21 to get the same results with matlab
         
    mask = torch.logical_and(torch.max(ratio_A2B, ratio_B2A) < ratio_th,  (ind_B[ind_A] == ind).view(num_input_points, -1))
    
    # set a large SSE score for mathces above ratio threshold and not on to one (score_A2B <=4 so use 5)
    score_A2B[~mask] = 5
    
    # each input point can generate max two output points, so discard the two with highest SSE 
    _, discard = torch.topk(score_A2B, 2, dim=1)
    
    mask[torch.arange(num_input_points), discard[:, 0]] = 0
    mask[torch.arange(num_input_points), discard[:, 1]] = 0
    
    # x & y coordiates of candidate match points of A
    x = points_A[0, :].repeat(4, 1).t() + neighbors[:, 0].repeat(num_input_points, 1)
    y = points_A[1, :].repeat(4, 1).t() + neighbors[: ,1].repeat(num_input_points, 1)
    
    refined_points_A = torch.stack((x[mask], y[mask]))
    
    # x & y coordiates of candidate match points of A
    x = points_B[0, :].repeat(4, 1).t() + neighbors[:, 0][match_A2B]
    y = points_B[1, :].repeat(4, 1).t() + neighbors[:, 1][match_A2B]
    
    refined_points_B = torch.stack((x[mask], y[mask]))
        
    # if the number of refined matches is not enough to estimate homography,
    # but number of initial matches is enough, use initial points
    if refined_points_A.shape[1] < 4 and num_input_points > refined_points_A.shape[1]:
        refined_points_A = points_A
        refined_points_B = points_B
    
    return refined_points_A, refined_points_B
    
def mnn_ratio_matcher(descriptors1, descriptors2, ratio=0.8, bidirectional = True):
    
    # Mutual NN + symmetric Lowe's ratio test matcher for L2 normalized descriptors.
    device = descriptors1.device
    sim = descriptors1 @ descriptors2.t()

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim, nns = torch.topk(sim, 2, dim=1)
    nns_dist = 2 - 2 * nns_sim
    # Compute Lowe's ratio.
    ratios12 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN and match similarity.
    nn12 = nns[:, 0]
    match_sim = nns_sim[:, 0]

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim, nns = torch.topk(sim.t(), 2, dim=1)
    nns_dist = 2 - 2 * nns_sim
    # Compute Lowe's ratio.
    ratios21 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN.
    nn21 = nns[:, 0]
    
    # if not bidirectional, do not use ratios from 2 to 1
    ratios21[:] *= 1 if bidirectional else 0
        
    # Mutual NN + symmetric ratio test.
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = torch.min(ids1 == nn21[nn12], torch.min(ratios12 <= ratio, ratios21[nn12] <= ratio)) # discard ratios21 to get the same results with matlab
    # Final matches.
    matches = torch.stack([ids1[mask], nn12[mask]], dim=-1)
    match_sim = match_sim[mask]

    return (matches.data.cpu().numpy(),match_sim.data.cpu().numpy())





















class Vgg19(torch.nn.Module):
    # modified from the original @ https://github.com/chenyuntc/pytorch-book/blob/master/chapter08-neural_style/PackedVGG.py
    def __init__(self, batch_normalization = True, required_layers = None):
        
        if required_layers == None and batch_normalization:
            required_layers = [3, 10, 17, 30, 43, 46]
        elif required_layers == None:
            required_layers = [2, 7, 12, 21, 30, 32]
            
        # features 2，7，12，21, 30, 32: conv1_2,conv2_2,relu3_2,relu4_2,conv5_2,conv5_3
        super(Vgg19, self).__init__()
        
        if batch_normalization:
            features = list(models.vgg19_bn(pretrained = True).features)[:47] # get vgg features
        else:
            features = list(models.vgg19(pretrained = True).features)[:33] # get vgg features
        
        self.features = torch.nn.ModuleList(features).eval() # construct network in eval mode
        
        for param in self.features.parameters(): # we don't need graidents, turn them of to save memory
            param.requires_grad = False
                
        self.required_layers = required_layers # record required layers to save them in forward
        
        for layer in required_layers[:-1]:
            self.features[layer+1].inplace = False # do not overwrite in any layer
        
    def forward(self, x):
        results = []
        for ii,model in enumerate(self.features):
            x = model(x)
            if ii in self.required_layers:
                results.append(x)
                
        vgg_outputs = namedtuple("VggOutputs", ['conv1_2', 'conv2_2', 'relu3_2', 'relu4_2', 'conv5_2', 'conv5_3'])
        
        return vgg_outputs(*results)



        