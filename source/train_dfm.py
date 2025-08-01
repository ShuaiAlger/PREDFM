#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 19:09:25 2021

@author: ufukefe
"""

import os
import argparse
import yaml
import cv2
import tqdm

from PIL import Image
import numpy as np
import time
import random

from TrainingParamMatcher import ParamMatcher
from train_utils import *



#To draw_matches
def draw_matches(img_A, img_B, keypoints0, keypoints1):
    
    p1s = []
    p2s = []
    dmatches = []
    for i, (x1, y1) in enumerate(keypoints0):
         
        p1s.append(cv2.KeyPoint(x1, y1, 1))
        p2s.append(cv2.KeyPoint(keypoints1[i][0], keypoints1[i][1], 1))
        j = len(p1s) - 1
        dmatches.append(cv2.DMatch(j, j, 1))
        
    matched_images = cv2.drawMatches(cv2.cvtColor(img_A, cv2.COLOR_RGB2BGR), p1s, 
                                     cv2.cvtColor(img_B, cv2.COLOR_RGB2BGR), p2s, dmatches, None)
    
    return matched_images


class TrainMatcher():
    def __init__(self, backbone):
        self.backbone = backbone
        with open(str("configs/"+backbone+".yml"), "r") as configfile:
            self.config = yaml.safe_load(configfile)['configuration']
        
        self.fm = ParamMatcher( enable_two_stage = self.config['enable_two_stage'],
                                model = self.config['model'], 
                                ratio_th = self.config['ratio_th'],
                                bidirectional = self.config['bidirectional'],
                                layers = self.config['layers'])

    def match(self, img_A, img_B):
        # H, H_init, points_A, points_B = self.fm.match(img_A, img_B)
        # return H, H_init, points_A, points_B
        points_A, points_B = self.fm.match(img_A, img_B)
        return points_A, points_B

    def generate_crops(self, image, rate: int):
        nums = rate ** 2
        h, w, c = image.shape
        kernel_h = h//rate
        kernel_w = w//rate
        image_crops = []
        for j in range(rate):
            for i in range(rate):
                image_crops.append(image[j*kernel_h:(j+1)*kernel_h, i*kernel_w:(i+1)*kernel_w, :])
        return image_crops

    def color_equalizeHist(self, image):
        b, g, r = cv2.split(image)
        
        b1 = cv2.equalizeHist(b)
        g1 = cv2.equalizeHist(g)
        r1 = cv2.equalizeHist(r)
        
        out = cv2.merge([b1,g1,r1])
        return out

    def make_random_transform(self, ref_image):
        h, w, c = ref_image.shape

        # half_h, half_w = h//2, w//2
        # image_crops = self.generate_crops(ref_image, 8)

        # target_image = ref_image
        kernel = (random.randint(1, 5)*2+1, random.randint(1, 5)*2+1)

        target_image = cv2.GaussianBlur(ref_image, kernel, 0)
        ref_image = cv2.GaussianBlur(ref_image, kernel, 0)

        do_equalize = random.randint(0, 1)
        target_image = self.color_equalizeHist(target_image) if do_equalize else target_image

        do_equalize = random.randint(0, 1)
        ref_image = self.color_equalizeHist(ref_image) if do_equalize else ref_image

        do_reverse = random.randint(0, 1)
        target_image = 255 - target_image if do_reverse else target_image

        do_reverse = random.randint(0, 1)
        ref_image = 255 - ref_image if do_reverse else ref_image


        tw = w
        th = h

        H = get_perspective_mat(
                                patch_ratio = 1, 
                                center_x = tw//2,
                                center_y = th//2,
                                pers_x = 0.0008,
                                pers_y = 0.0008,
                                shear_ratio = 0.10,
                                shear_angle = 10,
                                rotation_angle = 45,
                                scale = -0.4,
                                trans = 0.2
                                )

        warped_image = cv2.warpPerspective(target_image, H, (w, h))

        return target_image, warped_image, H

    def cal_homographies(self, p1s, p2s, w, h, H_gt):
        try:
            H_pred, inliers = cv2.findHomography(p1s, p2s, cv2.USAC_MAGSAC, ransacReprojThreshold=3, maxIters=5000, confidence=0.9999)
        except:
            H_pred = None
            inliers = np.zeros(0)

        if H_pred is None:
            correctness = np.zeros(10)
        else:
            corners = np.array([[ 0,         0, 1 ],
                                [ 0,     w - 1, 1 ],
                                [ h - 1,     0, 1 ],
                                [ h - 1, w - 1, 1 ]])
            real_warped_corners = np.dot(corners, np.transpose(H_gt))
            real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
            warped_corners = np.dot(corners, np.transpose(H_pred))
            warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
            mean_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
            correctness = np.array([float(mean_dist <= i) for i in range (1, 11)])
        return correctness

    def cal_MMAs(self, _m_pts1, _m_pts2, _H_gt):
        if (_m_pts1.shape[0] < 2) or (_m_pts1.shape[1] < 2):
            return np.zeros(10)
        else:
            MMAs = np.zeros(10)
            for thres in range(1, 11):
                t_ = thres*thres
                N_ = len(_m_pts1)
                sum_value_ = 0
                for i in range(N_):
                    new_pt = _H_gt @ np.array([_m_pts1[i][0], _m_pts1[i][1], 1])
                    new_pt /= new_pt[2]
                    du = new_pt[0] - _m_pts2[i][0]
                    dv = new_pt[1] - _m_pts2[i][1]
                    if (du*du + dv*dv) < t_:
                        sum_value_ = sum_value_ + 1
                MMAs[thres-1] = sum_value_/N_ if sum_value_ > 0 else 0.0
            return MMAs

    def get_ref_images(self):
        images = []
        for i in range(1):
            image = cv2.imread("test"+str(i)+".jpg")
            image = cv2.resize(image, (512, 512))
            images.append(image)
        return images

    def generate_train_data(self, generate_nums=50):
        ref_images = self.get_ref_images()
        imgs = []
        warped_imgs = []
        H_gts = []
        for i in range(len(ref_images)):
            origin_image = ref_images[i]
            for j in range(generate_nums):
                img, warped_img, H_gt = self.make_random_transform(origin_image)
                imgs.append(img)
                warped_imgs.append(warped_img)
                H_gts.append(H_gt)
        return imgs, warped_imgs, H_gts



    def matching_with_pics(self, imgs, warped_imgs, H_gts):
        error = 0.0

        for i in tqdm.trange(len(imgs)):
            canvas = np.concatenate([imgs[i], warped_imgs[i]], axis=1)
            cv2.imwrite("results/canvas"+str(i)+".jpg", canvas)

            H, H_init, p1s, p2s = self.match(imgs[i], warped_imgs[i])
            h, w, c = imgs[i].shape
            correctness = self.cal_homographies(p1s, p2s, w, h, H_gts[i])            
            error_h = (1 - correctness.sum()/10)/len(imgs)
            mmas = self.cal_MMAs(p1s, p2s, H_gts[i])
            error_mma = (1 - mmas.sum()/10)/len(imgs)
            error = error + 0.5*(error_h + error_mma)
            # error = error + error_h
        return error



    def testing(self):
        combination_nums = self.fm.get_combination_nums()
        from eval_megadepth_scannet import evaluate_megadepth, evaluate_scannet
        for i in tqdm.trange(combination_nums):
            self.fm.set_combination(i)
            print(self.fm.get_combination_names(i))
            evaluate_megadepth(self.match)



    def training(self):
        combination_nums = self.fm.get_combination_nums()

        imgs, warped_imgs, H_gts = self.generate_train_data()

        first_error_list = []
        for i in tqdm.trange(combination_nums):
            self.fm.set_combination(i)
            error = self.matching_with_pics(imgs, warped_imgs, H_gts)
            print("i: ", i, "  , error:  ", error)
            first_error_list.append(error)

        first_sorted_id = sorted(range(len(first_error_list)), key=lambda k: first_error_list[k], reverse=False)

        print([first_error_list[id] for id in first_sorted_id])

        error_list = []
        imgs, warped_imgs, H_gts = self.generate_train_data()
        for i in tqdm.trange(20):
            self.fm.set_combination(first_sorted_id[i])
            error = self.matching_with_pics(imgs, warped_imgs, H_gts)
            error_list.append(error)

        sorted_id = sorted(range(len(error_list)), key=lambda k: error_list[k], reverse=False)

        results_id = [first_sorted_id[id] for id in sorted_id]

        results = [self.fm.get_combination_names(index) for index in results_id]

        print([error_list[id] for id in sorted_id])
        for i in range(len(results)):
            print(results[i])
        print("Finish Training")



#Take arguments and configurations
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Algorithm wrapper for Image Matching Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_pairs', type=str, default='image_pairs.txt')
    args = parser.parse_args() 

    # with open("config.yml", "r") as configfile:
    #     config = yaml.safe_load(configfile)['configuration']

    # Make result directory
    # os.makedirs(config['output_directory'], exist_ok=True)     


    tm = TrainMatcher("vgg19")

    tm.testing()


