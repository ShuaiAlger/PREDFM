#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import torch
from torchvision import models, transforms
from collections import namedtuple
import matplotlib.pyplot as plt

import cv2

from se_utils import DescGroupPoolandNorm


import lightning as L

from torch import Tensor

from equinetworks.models import get_model


args = {
    "num_group": 8,
}
pool_and_norm = DescGroupPoolandNorm(args)



from get_kpts_desc import SuperPoint_extrator
spe = SuperPoint_extrator()




class FeatureExtractorModule(L.LightningModule):
    def __init__(
        self,
        model_name: str,
        fast_model: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = get_model(model_name, fixed_params=not fast_model, pretrained=True)


    def forward(self, x: Tensor) -> Tensor:
        return self.model.forward_multi_features(x)


    def load_state_dict(self, state_dict) -> None:
        d = {k[len("model.") :]: v for k, v in state_dict.items()}
        self.model.load_state_dict(d, strict=False)



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










class PREDFMatcher(torch.nn.Module):
    def __init__(self, model: str = 'c4resnet18',

                DETECTOR = 0, # # 0 is FAST, 1 is Shi-Tomasi, 2 is SuperPoint
                
                USE_FEATURE_SCALE = 4,
                USE_IMAGE_SCALE = 4,
                USE_SHIFT = 1

                 ):
        super(PREDFMatcher, self).__init__()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print("******   USE PREDFM   " + model + "*******")

        self.model = FeatureExtractorModule(model).cuda()

        self.DETECTOR = DETECTOR

        self.USE_FEATURE_SCALE = USE_FEATURE_SCALE
        self.USE_IMAGE_SCALE = USE_IMAGE_SCALE
        self.USE_SHIFT = USE_SHIFT


        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.padding_n = 16




    def extract_singlescale(self, img_A, img_B):
        grayA = cv2.cvtColor(img_A, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(img_B, cv2.COLOR_BGR2GRAY)
        
        cornersA = None
        cornersB = None


        # 0 is FAST, 1 is Shi-Tomasi, 2 is SuperPoint
        if self.DETECTOR == 0:
            fast = cv2.FastFeatureDetector_create(threshold=9, nonmaxSuppression=True)

            # find and draw the keypoints
            fastA = fast.detect(grayA, None)
            fastB = fast.detect(grayB, None)

            cornersA = np.zeros((len(fastA), 1, 2))
            cornersB = np.zeros((len(fastB), 1, 2))

            for i in range(len(fastA)):
                cornersA[i, 0, 0] = fastA[i].pt[0]
                cornersA[i, 0, 1] = fastA[i].pt[1]

            for i in range(len(fastB)):
                cornersB[i, 0, 0] = fastB[i].pt[0]
                cornersB[i, 0, 1] = fastB[i].pt[1]

        if self.DETECTOR == 1:
            grayA = np.float32(grayA)
            grayB = np.float32(grayB)
            cornersA = cv2.goodFeaturesToTrack(grayA, 10000, 0.03, 5)
            cornersB = cv2.goodFeaturesToTrack(grayB, 10000, 0.03, 5)
            # cornersA = cv2.goodFeaturesToTrack(grayA, 10000, 0.02, 4)
            # cornersB = cv2.goodFeaturesToTrack(grayB, 10000, 0.02, 4)

        if self.DETECTOR == 2:
            cornersA = spe(grayA)
            cornersB = spe(grayB)
            

        img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)
        img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)

        inp_A, padding_A = self.transform(img_A) 
        inp_B, padding_B = self.transform(img_B)

        outputs_A = self.model(inp_A)
        outputs_B = self.model(inp_B)

        activations_A = []
        activations_B = []


        for x in outputs_A:
            activations_A.append(x.tensor)
        for x in outputs_B:
            activations_B.append(x.tensor)

        for i in range(len(activations_A)):
            activations_A[i] = self.upsample(activations_A[i])
            activations_B[i] = self.upsample(activations_B[i])



        if (self.DETECTOR == 0) or (self.DETECTOR == 1):
            detectionA = np.roll(cornersA[:, 0, :], 1, axis=-1)
            detectionB = np.roll(cornersB[:, 0, :], 1, axis=-1)
        if self.DETECTOR == 2:
            detectionA = cornersA[:, 0:2]
            detectionB = cornersB[:, 0:2]



        descriptorsA = activations_A[0][0][:, detectionA[:, 0], detectionA[:, 1]]
        descriptorsB = activations_B[0][0][:, detectionB[:, 0], detectionB[:, 1]]

        if self.USE_FEATURE_SCALE > 1:
            descriptorsA = torch.concat([descriptorsA, activations_A[1][0][0:, np.int64(detectionA[:, 0]/2), np.int64(detectionA[:, 1]/2)]], dim=0)
            descriptorsB = torch.concat([descriptorsB, activations_B[1][0][0:, np.int64(detectionB[:, 0]/2), np.int64(detectionB[:, 1]/2)]], dim=0)
        if self.USE_FEATURE_SCALE > 2:
            descriptorsA = torch.concat([descriptorsA, activations_A[2][0][0:, np.int64(detectionA[:, 0]/4), np.int64(detectionA[:, 1]/4)]], dim=0)
            descriptorsB = torch.concat([descriptorsB, activations_B[2][0][0:, np.int64(detectionB[:, 0]/4), np.int64(detectionB[:, 1]/4)]], dim=0)
        if self.USE_FEATURE_SCALE > 3:
            descriptorsA = torch.concat([descriptorsA, activations_A[3][0][0:, np.int64(detectionA[:, 0]/8), np.int64(detectionA[:, 1]/8)]], dim=0)
            descriptorsB = torch.concat([descriptorsB, activations_B[3][0][0:, np.int64(detectionB[:, 0]/8), np.int64(detectionB[:, 1]/8)]], dim=0)
        if self.USE_FEATURE_SCALE > 4:
            descriptorsA = torch.concat([descriptorsA, activations_A[4][0][0:, np.int64(detectionA[:, 0]/16), np.int64(detectionA[:, 1]/16)]], dim=0)
            descriptorsB = torch.concat([descriptorsB, activations_B[4][0][0:, np.int64(detectionB[:, 0]/16), np.int64(detectionB[:, 1]/16)]], dim=0)


        return detectionA, detectionB, descriptorsA, descriptorsB

  


    def match_multiscale(self, img_A, img_B):
        maphA, mapwA = img_A.shape[0], img_A.shape[1]
        maphB, mapwB = img_B.shape[0], img_B.shape[1]

        imagesA, scalesA = adaptive_image_pyramid(img_A, max_size=1200)
        imagesB, scalesB = adaptive_image_pyramid(img_B, max_size=1200)

        # print(scalesA) # 1.4285, 1.2013, 1.0102, 0.8494, 0.7143, 0.6006, 0.5051, 0.4247, 0.3571

        pointsAs = []
        pointsBs = []
        descriptorsAs = []
        descriptorsBs = []

        for img_A, img_B, scale_A, scale_B in zip(imagesA, imagesB, scalesA, scalesB):
            pointsA, pointsB, descriptorsA, descriptorsB = self.extract_singlescale(img_A, img_B)
            pointsA = pointsA / scale_A
            pointsB = pointsB / scale_B


            if self.USE_SHIFT:
                descriptorsA = torch.tensor(descriptorsA)
                descriptorsA = torch.transpose(descriptorsA, 0, 1)
                descriptorsA = descriptorsA.unsqueeze(0)
                pointsA = torch.tensor(pointsA)
                pointsA = torch.transpose(pointsA, 0, 1)
                pointsA, descriptorsA = pool_and_norm.desc_pool_and_norm_infer(pointsA, descriptorsA)
                descriptorsA = torch.tensor(descriptorsA).squeeze(0)
                pointsA = pointsA.squeeze(0)
                pointsA = torch.transpose(pointsA, 0, 1).cpu().numpy()
                descriptorsA = torch.transpose(descriptorsA, 0, 1)

                descriptorsB = torch.tensor(descriptorsB)
                descriptorsB = torch.transpose(descriptorsB, 0, 1)
                descriptorsB = descriptorsB.unsqueeze(0)
                pointsB = torch.tensor(pointsB)
                pointsB = torch.transpose(pointsB, 0, 1)
                pointsB, descriptorsB = pool_and_norm.desc_pool_and_norm_infer(pointsB, descriptorsB)
                descriptorsB = torch.tensor(descriptorsB).squeeze(0)
                pointsB = pointsB.squeeze(0)
                pointsB = torch.transpose(pointsB, 0, 1).cpu().numpy()
                descriptorsB = torch.transpose(descriptorsB, 0, 1)

                # print(descriptorsA.shape)



            pointsAs.append(pointsA)
            pointsBs.append(pointsB)
            descriptorsAs.append(descriptorsA)
            descriptorsBs.append(descriptorsB)

        if self.USE_IMAGE_SCALE == 1:
            pointsAall = np.concatenate([pointsAs[6]], axis=0)
            pointsBall = np.concatenate([pointsBs[6]], axis=0)
            descriptorsAsall = torch.cat([descriptorsAs[6]], dim=1)
            descriptorsBsall = torch.cat([descriptorsBs[6]], dim=1)
        if self.USE_IMAGE_SCALE == 2:
            pointsAall = np.concatenate([pointsAs[2], pointsAs[6]], axis=0)
            pointsBall = np.concatenate([pointsBs[2], pointsBs[6]], axis=0)
            descriptorsAsall = torch.cat([descriptorsAs[2], descriptorsAs[6]], dim=1)
            descriptorsBsall = torch.cat([descriptorsBs[2], descriptorsBs[6]], dim=1)
        if self.USE_IMAGE_SCALE == 3:
            pointsAall = np.concatenate([pointsAs[0], pointsAs[3], pointsAs[7]], axis=0)
            pointsBall = np.concatenate([pointsBs[0], pointsBs[3], pointsBs[7]], axis=0)
            descriptorsAsall = torch.cat([descriptorsAs[0], descriptorsAs[3], descriptorsAs[7]], dim=1)
            descriptorsBsall = torch.cat([descriptorsBs[0], descriptorsBs[3], descriptorsBs[7]], dim=1)
        if self.USE_IMAGE_SCALE == 4:
            pointsAall = np.concatenate([pointsAs[0], pointsAs[2], pointsAs[5], pointsAs[7]], axis=0)
            pointsBall = np.concatenate([pointsBs[0], pointsBs[2], pointsBs[5], pointsBs[7]], axis=0)
            descriptorsAsall = torch.cat([descriptorsAs[0], descriptorsAs[2], descriptorsAs[5], descriptorsAs[7]], dim=1)
            descriptorsBsall = torch.cat([descriptorsBs[0], descriptorsBs[2], descriptorsBs[5], descriptorsBs[7]], dim=1)



        if self.USE_IMAGE_SCALE == 8:
            pointsAall = np.concatenate([pointsAs[0], pointsAs[2], pointsAs[5], pointsAs[7],
                                        pointsAs[1], pointsAs[3], pointsAs[4], pointsAs[6]], axis=0)
            pointsBall = np.concatenate([pointsBs[0], pointsBs[2], pointsBs[5], pointsBs[7],
                                        pointsBs[1], pointsBs[3], pointsBs[4], pointsBs[6]], axis=0)
            descriptorsAsall = torch.cat([descriptorsAs[0], descriptorsAs[2], descriptorsAs[5], descriptorsAs[7],
                                        descriptorsAs[1], descriptorsAs[3], descriptorsAs[4], descriptorsAs[6]], dim=1)
            descriptorsBsall = torch.cat([descriptorsBs[0], descriptorsBs[2], descriptorsBs[5], descriptorsBs[7],
                                        descriptorsBs[1], descriptorsBs[3], descriptorsBs[4], descriptorsBs[6]], dim=1)

        matches, scores = dense_feature_matching(descriptorsAsall, descriptorsBsall, 0.999, True)

        # matches, scores = dense_feature_matching(descriptorsAsall, descriptorsBsall, 0.7, True)

        pointsA = pointsAall[matches[:, 0]]
        pointsB = pointsBall[matches[:, 1]]

        pointsA = np.roll(pointsA, 1, axis=-1)
        pointsB = np.roll(pointsB, 1, axis=-1)

        return pointsA, pointsB




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

    d1 = map_A.t()
    d1 /= torch.sqrt(torch.sum(torch.square(d1), 1)).unsqueeze(1)
    
    d2 = map_B.t()
    d2 /= torch.sqrt(torch.sum(torch.square(d2), 1)).unsqueeze(1)

    # perform matching
    matches, scores = mnn_ratio_matcher(d1, d2, ratio_th, bidirectional)

    return matches, scores





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

