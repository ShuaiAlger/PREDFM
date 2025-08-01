import argparse
from pathlib import Path
from typing import List, Tuple

import lightning as L
import torch

from torch import Tensor

from equivision.models import get_model


import numpy as np

import cv2





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


def main(hparams):
    # create run name
    L.seed_everything(hparams.seed, workers=True)

    model = FeatureExtractorModule(
        model_name=hparams.model
        )

    with torch.no_grad():

        raw_image = cv2.imread("0a054772265311c3e7b3ac8fc29c54.jpg")

        input_demo = torch.tensor(raw_image)/255.0

        input_demo = input_demo.transpose(0, 2).transpose(1, 2).unsqueeze(0)

        outputs = model(input_demo)

        for o in outputs:
            print(o.shape)

        output = outputs[2]

        out_featuremap = output.tensor.detach().numpy()


        out_img0 = cv2.normalize(out_featuremap[0, 0], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        out_img1 = cv2.normalize(out_featuremap[0, 1], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        out_img2 = cv2.normalize(out_featuremap[0, 2], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        out_img3 = cv2.normalize(out_featuremap[0, 3], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

        canvas1 = np.concatenate([out_img0, out_img1, out_img2, out_img3], axis=1)

        canvas1 = cv2.applyColorMap(canvas1, cv2.COLORMAP_JET)

        cv2.imshow("canvas1", canvas1)


        rot_image = cv2.rotate(raw_image, cv2.ROTATE_90_CLOCKWISE)

        input_demo = torch.tensor(rot_image)/255.0

        input_demo = input_demo.transpose(0, 2).transpose(1, 2).unsqueeze(0)

        outputs = model(input_demo)

        output = outputs[2]

        out_featuremap = output.tensor.detach().numpy()

        print("out_featuremap.shape : ", out_featuremap.shape)


        out_img0_rot = cv2.normalize(out_featuremap[0, 0], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        out_img1_rot = cv2.normalize(out_featuremap[0, 1], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        out_img2_rot = cv2.normalize(out_featuremap[0, 2], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        out_img3_rot = cv2.normalize(out_featuremap[0, 3], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

        out_img0_rot = cv2.rotate(out_img0_rot, cv2.ROTATE_90_COUNTERCLOCKWISE)
        out_img1_rot = cv2.rotate(out_img1_rot, cv2.ROTATE_90_COUNTERCLOCKWISE)
        out_img2_rot = cv2.rotate(out_img2_rot, cv2.ROTATE_90_COUNTERCLOCKWISE)
        out_img3_rot = cv2.rotate(out_img3_rot, cv2.ROTATE_90_COUNTERCLOCKWISE)

        canvas2 = np.concatenate([out_img0_rot, out_img1_rot, out_img2_rot, out_img3_rot], axis=1)

        canvas2 = cv2.applyColorMap(canvas2, cv2.COLORMAP_JET)

        cv2.imshow("canvas2", canvas2)


        cv2.imwrite("origin_image_input.png", canvas1)
        cv2.imwrite("rot90_image_input.png", canvas2)



        rot_image = cv2.rotate(raw_image, cv2.ROTATE_180)

        input_demo = torch.tensor(rot_image)/255.0

        input_demo = input_demo.transpose(0, 2).transpose(1, 2).unsqueeze(0)

        outputs = model(input_demo)

        output = outputs[2]

        out_featuremap = output.tensor.detach().numpy()

        print("out_featuremap.shape : ", out_featuremap.shape)


        out_img0_rot = cv2.normalize(out_featuremap[0, 0], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        out_img1_rot = cv2.normalize(out_featuremap[0, 1], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        out_img2_rot = cv2.normalize(out_featuremap[0, 2], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        out_img3_rot = cv2.normalize(out_featuremap[0, 3], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

        out_img0_rot = cv2.rotate(out_img0_rot, cv2.ROTATE_180)
        out_img1_rot = cv2.rotate(out_img1_rot, cv2.ROTATE_180)
        out_img2_rot = cv2.rotate(out_img2_rot, cv2.ROTATE_180)
        out_img3_rot = cv2.rotate(out_img3_rot, cv2.ROTATE_180)

        canvas2 = np.concatenate([out_img0_rot, out_img1_rot, out_img2_rot, out_img3_rot], axis=1)

        canvas2 = cv2.applyColorMap(canvas2, cv2.COLORMAP_JET)


        cv2.imwrite("rot180_image_input.png", canvas2)




        rot_image = cv2.rotate(raw_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        input_demo = torch.tensor(rot_image)/255.0

        input_demo = input_demo.transpose(0, 2).transpose(1, 2).unsqueeze(0)

        outputs = model(input_demo)

        output = outputs[2]

        out_featuremap = output.tensor.detach().numpy()

        print("out_featuremap.shape : ", out_featuremap.shape)


        out_img0_rot = cv2.normalize(out_featuremap[0, 0], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        out_img1_rot = cv2.normalize(out_featuremap[0, 1], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        out_img2_rot = cv2.normalize(out_featuremap[0, 2], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        out_img3_rot = cv2.normalize(out_featuremap[0, 3], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

        out_img0_rot = cv2.rotate(out_img0_rot, cv2.ROTATE_90_CLOCKWISE)
        out_img1_rot = cv2.rotate(out_img1_rot, cv2.ROTATE_90_CLOCKWISE)
        out_img2_rot = cv2.rotate(out_img2_rot, cv2.ROTATE_90_CLOCKWISE)
        out_img3_rot = cv2.rotate(out_img3_rot, cv2.ROTATE_90_CLOCKWISE)

        canvas2 = np.concatenate([out_img0_rot, out_img1_rot, out_img2_rot, out_img3_rot], axis=1)

        canvas2 = cv2.applyColorMap(canvas2, cv2.COLORMAP_JET)


        cv2.imwrite("rot270_image_input.png", canvas2)






        cv2.waitKey(0)





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)




    parser.add_argument("--model", type=str, default="c4resnet18")


    args = parser.parse_args()


    main(args)









