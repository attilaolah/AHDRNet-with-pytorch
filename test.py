"""Tests.
Author: Wei Wang
"""
import os

import cv2
import math
import numpy as np
import torch

import models
import utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


PIXEL_MAX = 1


def psnr2(img1, img2):
    mse = np.mean((img1/255. - img2/255.) ** 2)
    if mse < 1.0e-10:
        return 100
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


scene_directory = os.path.join('data', 'Test', 'PAPER', 'ManStanding')


def inference() -> None:
    # Load the image
    # Read Expo times in scene
    expoTimes = utils.ReadExpoTimes(os.path.join(scene_directory,
                                                 'exposure.txt'))
    # Read Image in scene
    imgs = utils.ReadImages(utils.list_all_files_sorted(scene_directory,
                                                        '.tif'))
    # Read label
    label = utils.ReadLabel(scene_directory)
    # inputs-process
    pre_img0 = utils.LDR_to_HDR(imgs[0], expoTimes[0], 2.2)
    pre_img1 = utils.LDR_to_HDR(imgs[1], expoTimes[1], 2.2)
    pre_img2 = utils.LDR_to_HDR(imgs[2], expoTimes[2], 2.2)
    output0 = np.concatenate((imgs[0], pre_img0), 2)
    output1 = np.concatenate((imgs[1], pre_img1), 2)
    output2 = np.concatenate((imgs[2], pre_img2), 2)
    # label-process
    label = utils.range_compressor(label)*255.0

    im1 = torch.Tensor(output0).cuda()
    im1 = torch.unsqueeze(im1, 0).permute(0, 3, 1, 2)

    im2 = torch.Tensor(output1).cuda()
    im2 = torch.unsqueeze(im2, 0).permute(0, 3, 1, 2)

    im3 = torch.Tensor(output2).cuda()
    im3 = torch.unsqueeze(im3, 0).permute(0, 3, 1, 2)

    # Load the pre-trained model
    model = models.AHDRNet().cuda()
    model.eval()
    model.load_state_dict(torch.load('train_results/model/latest.pkl')['model'])

    # Run
    with torch.no_grad():
        # Forward
        pre = model(im1, im2, im3)
    pre = torch.clamp(pre, 0., 1.)
    pre = pre.data[0].cpu().numpy()
    pre = np.clip(pre * 255.0, 0., 255.)
    pre = pre.transpose(1, 2, 0)
    p = psnr2(pre, label)
    print(p)
    cv2.imwrite('./recover/PeopleTalking/out.png', pre)
    cv2.imwrite('./recover/PeopleTalking/label.png', label)


if __name__ == '__main__':
    inference()
