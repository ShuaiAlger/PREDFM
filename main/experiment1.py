# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import mmcv
from mmdet.apis import inference_detector, init_detector

from mmrotate.registry import VISUALIZERS
from mmrotate.utils import register_all_modules

import glob
import tqdm
import cv2
import torch
from torchvision import transforms
import matplotlib.pyplot as plt


import copy

def draw_green(_img1, _img2, _m_pts1_src, _m_pts2_src):
    canvas = 255*np.ones((max(_img1.shape[0],_img2.shape[0]), _img1.shape[1]+_img2.shape[1], 3), dtype=np.uint8)

    _m_pts1 = copy.deepcopy(_m_pts1_src.copy())
    _m_pts2 = copy.deepcopy(_m_pts2_src.copy())

    canvas[0:_img1.shape[0], 0:_img1.shape[1], :] = _img1
    canvas[0:_img2.shape[0], _img1.shape[1]:_img1.shape[1]+_img2.shape[1], :] = _img2
    
    N_ = len(_m_pts1)

    for i in range(N_):
        color = (0, 255, 0)
        cv2.line(canvas, (int(_m_pts1[i, 0]), int(_m_pts1[i, 1])), (int(_m_pts2[i, 0]+_img1.shape[1]), int(_m_pts2[i, 1])), color, 1)
    return canvas


def desc2hist(desc):
    hist_A = np.asarray(desc * 255, dtype=np.uint8)
    hist_A_img = np.zeros((hist_A.shape[0], 256), dtype=np.uint8)
    for i in range(hist_A.shape[0]):
        hist_A_img[i, 255-hist_A[i]:255] = 255
    hist_A_img = hist_A_img.T
    return hist_A_img


def adaptive_image_pyramid(img, min_scale=0.0, max_scale=1, min_size=256, max_size=1536, scale_f=2**0.25, verbose=False):
    
    H, W, C = img.shape

    ## upsample the input to bigger size.
    s = 1.0
    if max(H, W) < max_size:
        s = max_size / max(H, W)
        max_scale = s
        nh, nw = round(H*s), round(W*s)
        # if verbose:  print(f"extracting at highest scale x{s:.02f} = {nw:4d}x{nh:3d}")
        # img = F.interpolate(img, (nh,nw), mode='bilinear', align_corners=False)

        img = cv2.resize(img, (nw, nh))
        
    ## downsample the scale pyramid
    output = []
    scales = []
    while s+0.001 >= max(min_scale, min_size / max(H,W)):
        if s-0.001 <= min(max_scale, max_size / max(H,W)):
            nh, nw = img.shape[0:2]

            if verbose: print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")
            output.append(img)
            scales.append(s)
        # print(f"passing the loop x{s:.02f} = {nw:4d}x{nh:3d}")        

        s /= scale_f

        # down-scale the image for next iteration
        nh, nw = round(H*s), round(W*s)
        img = cv2.resize(img, (nw, nh))
    
    return output, scales



import torch
import torch.nn as nn


def shift_topk_candidate(kpts, desc, B, K, G, topk=1):
    shifts= torch.topk(desc[:,:,0,:], k= topk, dim=2)[1]
    desc = desc.reshape(B*K, -1, G)
    shifts = shifts.reshape(B*K, -1)

    kpts_update = kpts.repeat(1, topk, 1) ## [B, topk*K, 2] 
    desc_update = []
    for shift in shifts.t():
        for d, s in zip(desc, shift):
            desc_update.append(torch.roll(d, shifts=-int(s), dims=-1)) ## reverse shift

    desc_update = torch.stack(desc_update) ## [topk*B*K, C, G]
    desc_update = desc_update.reshape(topk,B,K,-1,G).transpose(0,1) ## [B*topk*K, C, G]   
    desc_update = desc_update.reshape(B, topk*K,-1)
    return kpts_update, desc_update

def shift_ratio_candidate(kpts, desc, B, K, CG, G, ratio=1.0):
    ## Warning: ratio sampling make different number of keypoints in an image.
    assert desc.min() >= 0
    value, _ = torch.max(desc[:, :, 0, :], dim=2)
    ratio_tensor = (desc[:, :, 0, :] / value.unsqueeze(-1))  ## obtain ratio
    ratio_mask = ratio_tensor >= ratio

    ### un-batchfying because the number of keypoints are different.
    kpts_update = []
    desc_update = []
    for _kpts, _desc, _ratio_mask in zip(kpts, desc, ratio_mask):
        kpts_update_iter = []
        desc_update_iter = []
        for k, d, r in zip(_kpts, _desc, _ratio_mask):
            shifts = r.nonzero().reshape(-1)
            for s in shifts:
                kpts_update_iter.append(k)
                desc_update_iter.append(torch.roll(d, shifts=-int(s), dims=-1))
            
        kpts_update.append(torch.stack(kpts_update_iter))
        desc_update.append(torch.stack(desc_update_iter).reshape(-1, CG))
    return kpts_update, desc_update


def shifting(desc, shift, B, K, G):
    desc = desc.reshape(B*K, -1, G)
    shift = shift.reshape(B*K)
    
    desc_update = []
    for d, s in zip(desc, shift):
        desc_update.append(torch.roll(d, shifts=-int(s), dims=-1)) ## reverse shift
    desc_update = torch.stack(desc_update).reshape(B,K,-1)
    return desc_update


class DescGroupPoolandNorm:
    def __init__(self, args):
        self.pool = 'shift'
        self.norm = "l2"
        self.candidate = "top1"

        self.G = args['num_group']


    def desc_pool_and_norm_infer(self, kpts, desc):

        """ kpts [B, K, 2], desc torch.tensor([B, K, CG])
        """
        B, K, CG = desc.shape
        desc = desc.reshape(B,K,-1,self.G)

        if self.pool == 'shift':
            if 'top' in self.candidate:
                topk = int(self.candidate[3])
                kpts_update, desc_update = shift_topk_candidate(kpts, desc, B, K, self.G, topk)
            else:  ## ratio
                ratio = float(self.candidate)
                kpts_update, desc_update = shift_ratio_candidate(kpts, desc, B, K, CG, self.G, ratio)
        else:
            kpts_update = kpts
            desc_update = self._descriptor_pool(desc)

        if 'top' in self.candidate or self.pool!='shift': ## batchified version
            desc_update = self._descriptor_norm(desc_update)
        else: ## ratio (batch as list)
            desc_update_list = []
            for dd in desc_update:
                desc_update_list.append( self._descriptor_norm(dd))
            desc_update = desc_update_list
        
        return kpts_update, desc_update

    def desc_pool_and_norm(self, desc, GTShift):
        """ desc [B, K, CG], for training.
        """

        B, K, CG = desc.shape
        desc = desc.reshape(B,K,-1,self.G)

        if self.pool=='shift':
            shift = GTShift.unsqueeze(1).repeat(1, K)       
            desc_update = shifting(desc, shift, B, K, self.G)
        else:
            desc_update = self._descriptor_pool(desc)

        desc_update = self._descriptor_norm(desc_update)

        self._exception_handling(desc_update, desc, B, K, CG, shift)

        return desc_update

    def _descriptor_pool(self, desc):
        B, K, CG = desc.shape
        if self.pool=='avg':
            desc_update = desc.mean(dim=3)
        elif self.pool=='weight_avg':
            weight = desc[:, :, 0, :].unsqueeze(2) 
            desc_update = (weight * desc).mean(dim=3)
        elif self.pool=='max':
            desc_update = desc.max(dim=3)[0]
        elif self.pool=='norm':
            desc_update = desc.norm(dim=3)
        elif self.pool=='none':
            desc_update = desc.reshape(B, K, CG)
        else:
            raise NotImplementedError
        return desc_update

    def _descriptor_norm(self, desc_update):
        ## descriptor normalize; default: l2 normalize
        if self.norm=='l2':
            desc_update = L2Norm()(desc_update)
        elif self.norm =='l1':
            desc_update = L1Norm()(desc_update)
        elif self.norm =='context':
            desc_update = ContextNorm()(desc_update)
        elif self.norm == 'none':
            desc_update = desc_update
        else:
            raise NotImplementedError
        return desc_update


    def _exception_handling(self, desc_update, desc, B, K, CG, shift):
        if self.pool == 'shift' or self.pool =='shift_avg' or self.pool =='shift_all':
            assert desc_update.shape == desc.reshape(B,K,CG).shape
            assert shift.max() < self.G and shift.min() >=0
        elif  self.pool =='none':
            assert desc_update.shape == desc.reshape(B,K,CG).shape
        elif self.pool == 'bilinear':
            assert desc_update.shape[-1] == 128
        else:
            assert desc_update.shape[:3] == desc.shape[:3]  ## [B, K, C]


######## descriptor normalize

class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()
        self.eps = 1e-6

    def forward(self, x):
        """ x [B, K, CG] or [K, CG]"""
        # norm = torch.sqrt(torch.sum(x * x, dim=1) + self.eps)
        norm = torch.norm(x, p=2, dim=-1) + self.eps
        x = x / norm.unsqueeze(-1)
        return x

class L1Norm(nn.Module):
    def __init__(self):
        super(L1Norm, self).__init__()
        self.eps = 1e-6

    def forward(self, x):
        # norm = torch.sum(torch.abs(x), dim=1) + self.eps
        norm = torch.norm(x, p=1, dim=-1) + self.eps
        x = x / norm.unsqueeze(-1)
        return x

class ContextNorm(nn.Module):
    def __init__(self):
        super(ContextNorm, self).__init__()
        self.eps = 1e-6
    
    def forward(self, x):
        mean = torch.mean(x, dim=-1) 
        std = torch.std(x, dim=-1) + self.eps
        x = (x - mean.unsqueeze(-1)) / std.unsqueeze(-1)
        return x


pool_and_norm = DescGroupPoolandNorm({
                                        "num_group": 8,
                                    })



class REDFM(torch.nn.Module):
    def __init__(self,
                model,          
                USE_FEATURE_SCALE = 4,
                USE_SHIFT = 1
                 ):
        super(REDFM, self).__init__()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.extractor = model

        self.USE_FEATURE_SCALE = USE_FEATURE_SCALE
        self.USE_SHIFT = USE_SHIFT

        self.return_nodes = {
        }

        self.upsample = torch.nn.Upsample(scale_factor = 4, mode='bilinear', align_corners=True)
        self.padding_n = 16

        self.bidirectional = False

        # self.ratio_th = np.array([0.9, 0.95, 0.95, 0.999, 0.999, 0.999])

        self.ratio_th = np.array([0.9, 0.9, 0.9, 0.9, 1.0])

        # self.ratio_th = np.array([0.97, 1.0, 1.0, 1.0, 1.0])


    def padding_numpy(self, image, padding_size=16):
        h, w = image.shape[0], image.shape[1]
        padh = int(h % padding_size)
        padw = int(w % padding_size)

        image_new = np.pad(image, ((0, padh), (0, padw), (0, 0)), 'constant', constant_values=0)
        return image_new, (padw, padh)

    def match(self, img_A, img_B):

        '''
        H: homography matrix warps image B onto image A, compatible with cv2.warpPerspective,
        and match points on warped image = H * match points on image B
        H_init: stage 0 homography
        '''

        ### transform into pytroch tensor and pad image to a multiple of 16
        img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)
        img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)

        inp_A, padding_A = self.padding_numpy(img_A)
        inp_B, padding_B = self.padding_numpy(img_B)


        outputs_A = inference_detector(self.extractor, inp_A/255.0)
        outputs_B = inference_detector(self.extractor, inp_B/255.0)


        activations_A = []
        activations_B = []

        STAGES = 4

        for i in range(0, STAGES):
            activations_A.append(outputs_A[i].tensor)
        for i in range(0, STAGES):
            activations_B.append(outputs_B[i].tensor)

        for i in range(len(activations_A)):
            activations_A[i] = self.upsample(activations_A[i])
            activations_B[i] = self.upsample(activations_B[i])


        # print(inp_A.shape, activations_A[0].shape, " - ", activations_A[1].shape, activations_A[2].shape, activations_A[3].shape)


        # # initiate matches
        # points_A, points_B = self.dense_feature_matching(activations_A[-1], activations_B[-1], self.ratio_th[-1], self.bidirectional)

        # for k in range(len(activations_A) - 2, -1, -1):
        #     points_A, points_B = self.refine_points(points_A, points_B, activations_A[k], activations_B[k], self.ratio_th[k], self.bidirectional)


        # initiate matches
        points_A, points_B = self.dense_feature_matching(activations_A[-1], activations_B[-1], 1.0, self.bidirectional)

        points_A = points_A * 8
        points_B = points_B * 8

        points_A = points_A.double()
        points_B = points_B.double()

        # optional
        newHA, newWA = inp_A.shape[0], inp_A.shape[1]
        newHB, newWB = inp_B.shape[0], inp_B.shape[1]

        in_image = torch.logical_and(points_A[0, :] < (newWA - padding_A[0] - 16), points_A[1, :] < (newHA - padding_A[1] - 16))
        in_image = torch.logical_and(in_image, 
                                     torch.logical_and(points_B[0, :] < (newWB - padding_B[0] - 16), points_B[1, :] < (newHB - padding_B[1] - 16)))

        points_A = points_A[:, in_image]
        points_B = points_B[:, in_image]


        # return H, H_init, points_A.numpy(), points_B.numpy()
        return points_A.numpy().T, points_B.numpy().T




    def transform(self, img):

        '''
        Convert given uint8 numpy array to tensor, perform normalization and 
        pad right/bottom to make image canvas a multiple of self.padding_n

        Parameters
        ----------
        img : numpy array (uint8)

        Returns
        -------
        img_T : torch.tensor
        (pad_right, pad_bottom) : int tuple 

        '''
        
        # transform to tensor and normalize
        T = transforms.Compose([
                                transforms.ToTensor(),
                                # transforms.Lambda(lambda x: x.to(self.device)),
                                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                #                      std=[0.229, 0.224, 0.225])
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



    def dense_feature_matching(self, map_A, map_B, ratio_th, bidirectional=True):
        # normalize and reshape feature maps
        _, ch, h_A, w_A = map_A.size()
        _, _,  h_B, w_B = map_B.size()

        d1_ = map_A.view(ch, -1).t()
        d1 = d1_ / torch.sqrt(torch.sum(torch.square(d1_), 1)).unsqueeze(1)

        d2_ = map_B.view(ch, -1).t()
        d2 = d2_ / torch.sqrt(torch.sum(torch.square(d2_), 1)).unsqueeze(1)

        # form a coordinate grid and convert matching indexes to image coordinates
        y_A, x_A = torch.meshgrid(torch.arange(h_A), torch.arange(w_A))
        y_B, x_B = torch.meshgrid(torch.arange(h_B), torch.arange(w_B))

        if self.USE_SHIFT:

            init_points_A = torch.concat([x_A.reshape(w_A*h_A, 1), y_A.reshape(w_A*h_A, 1)], dim=1)
            init_points_B = torch.concat([x_B.reshape(w_B*h_B, 1), y_B.reshape(w_B*h_B, 1)], dim=1)


            d1 = d1.unsqueeze(0)

            print("d1.shape : ", d1.shape)

            init_points_A, d1 = pool_and_norm.desc_pool_and_norm_infer(init_points_A, d1)
            d1 = d1.squeeze(0)

            d2 = d2.unsqueeze(0)
            init_points_B, d2 = pool_and_norm.desc_pool_and_norm_infer(init_points_B, d2)
            d2 = d2.squeeze(0)

        # perform matching
        matches, scores = self.mnn_ratio_matcher(d1, d2, ratio_th, bidirectional)

        points_A = torch.stack((x_A.flatten()[matches[:, 0]], y_A.flatten()[matches[:, 0]]))
        points_B = torch.stack((x_B.flatten()[matches[:, 1]], y_B.flatten()[matches[:, 1]]))
        
        # discard the point on image boundaries
        discard = (points_A[0, :] == 0) | (points_A[0, :] == w_A-1) | (points_A[1, :] == 0) | (points_A[1, :] == h_A-1) \
                | (points_B[0, :] == 0) | (points_B[0, :] == w_B-1) | (points_B[1, :] == 0) | (points_B[1, :] == h_B-1)
        
        #discard[:] = False
        points_A = points_A[:, ~discard]
        points_B = points_B[:, ~discard]
        
        return points_A, points_B



    def refine_points(self, points_A: torch.Tensor, points_B: torch.Tensor, activations_A: torch.Tensor, activations_B: torch.Tensor, ratio_th = 0.9, bidirectional = True):

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


                if self.USE_SHIFT:
                
                    act_A = act_A.transpose(0, 1)
                    act_A = act_A.unsqueeze(0)
                    _, act_A = pool_and_norm.desc_pool_and_norm_infer(points_A[:, :].transpose(1, 0), act_A)
                    act_A = act_A.squeeze(0)
                    act_A = act_A.transpose(0, 1)

                    act_B = act_B.transpose(0, 1)
                    act_B = act_B.unsqueeze(0)
                    _, act_B = pool_and_norm.desc_pool_and_norm_infer(points_B[:, :].transpose(1, 0), act_B)
                    act_B = act_B.squeeze(0)
                    act_B = act_B.transpose(0, 1)

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



    def mnn_ratio_matcher(self, descriptors1, descriptors2, ratio=0.8, bidirectional = True):
        
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












#################################################################################################################









def parse_args():
    parser = ArgumentParser()
    # parser.add_argument('img', help='Image file')
    parser.add_argument('--config', default="configs/redet/redet-le90_re50_refpn_rr-1x_dota-ms.py", help='Config file')
    parser.add_argument('--checkpoint', default="redet_re50_fpn_1x_dota_ms_rr_le90-fc9217b5.pth", help='Checkpoint file')
    # parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='dota',
        choices=['dota', 'sar', 'hrsc', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args






def build_model(args):
    # register all modules in mmrotate into the registries
    register_all_modules()

    # build the model from a config file and a checkpoint file
    model = init_detector(
        args.config, args.checkpoint, palette=args.palette, device=args.device)
    # result = inference_detector(model, img)
    return model


def match_main(args):
    visloc_data_root = "/media/shuai/ssbigdisk/Datasets/UAV_VisLoc_dataset/01/drone/"
    frames_path = sorted(glob.glob(visloc_data_root+"/*.JPG"))

    model = build_model(args)

    matcher = REDFM(model, USE_FEATURE_SCALE=4, USE_SHIFT=0)

    for i in tqdm.trange(0, len(frames_path)-1):
        last_frame = cv2.imread(frames_path[i])
        curr_frame = cv2.imread(frames_path[i+1])

        matches, scores = matcher.match(last_frame, curr_frame)


def demo_camera_pose(function_match, USE_ROT90_DATASET=0, USE_CONTINUE_ROT_DATASET=0):
# eval megadepth
    from eval_megadepth_scannet_yfcc import evaluate_megadepth

    evaluate_megadepth(function_match,
                       USE_ROT90_DATASET= USE_ROT90_DATASET,
                       USE_CONTINUE_ROT_DATASET= USE_CONTINUE_ROT_DATASET
                       )


def eval_main(args):
    model = build_model(args)

    matcher = REDFM(model, USE_FEATURE_SCALE=4, USE_SHIFT=1)

    demo_camera_pose(matcher.match,
                USE_ROT90_DATASET = 0 ,
                USE_CONTINUE_ROT_DATASET = 0
                )



def main(args):
    # register all modules in mmrotate into the registries
    register_all_modules()

    # build the model from a config file and a checkpoint file
    model = init_detector(
        args.config, args.checkpoint, palette=args.palette, device=args.device)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta

    # test a single image
    result = inference_detector(model, args.img)

    # show the results
    img = mmcv.imread(args.img)
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    visualizer.add_datasample(
        'result',
        img,
        data_sample=result,
        draw_gt=False,
        show=args.out_file is None,
        wait_time=0,
        out_file=args.out_file,
        pred_score_thr=args.score_thr)

import pickle
import numpy as np

if __name__ == '__main__':
    args = parse_args()

    eval_main(args)
