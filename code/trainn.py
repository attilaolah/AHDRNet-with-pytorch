import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
import numpy as np
import torch
from torch import autograd
from torch import nn
from torch import optim
from tqdm import tqdm

from datsetprocess import *
from model import *
from opts import TrainOptions
from utils import *


"""
    Author: Wei Wang
"""


def train(opts):
    # Create the loader
    global ep
    train_data = MyDataset(scene_directory=opts.folder)
    loader = DataLoader(train_data, batch_size=opts.batch_size,
                        shuffle=True, num_workers=1)
    # Create the model
    model = AHDRNet().cuda()
    criterion = nn.L1Loss().to(opts.device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Load pre-train model
    if os.path.exists(opts.resume):
        state = torch.load(opts.resume)
        Loss_list = state['loss']
        model.load_state_dict(state['model'])
    else:
        Loss_list = []
    model.load_state_dict(torch.load('./Model/900.pkl'))
    # Train
    bar = tqdm(range(901, 15000))
    for ep in bar:
        loss_list = []
        for step, sample in enumerate(loader):
            (batch_x1, batch_x2, batch_x3, batch_x4) = (
                sample['input1'],
                sample['input2'],
                sample['input3'],
                sample['label'],
            )
            (batch_x1, batch_x2, batch_x3, batch_x4) = (
                autograd.Variable(batch_x1).cuda(),
                autograd.Variable(batch_x2).cuda(),
                autograd.Variable(batch_x3).cuda(),
                autograd.Variable(batch_x4).cuda(),
            )

            # Forward and compute loss
            pre = model(batch_x1, batch_x2, batch_x3)
            #pre = (torch.log(torch.FloatTensor(1).cuda() + 5000 * out)) / torch.log(torch.FloatTensor(1 + 5000).cuda())
            loss = criterion(pre, batch_x4)
            psnr = batch_PSNR(torch.clamp(pre, 0., 1.), batch_x4, 1.0)
            loss_list.append(loss.item())
            bar.set_description("Epoch: %d   Loss: %.6f" % (ep, loss_list[-1]))

            # Update the parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("[Epoch %d][G loss: %7f][PSNR : %7f]" %
                  (ep, loss_list[-1], psnr))
        Loss_list.append(np.mean(loss_list))

        # Save the training image
        '''
         if ep % opts.record_epoch == 0:
            img = fusePostProcess(y_f, y_hat, patch1, patch2, single = False)
            cv2.imwrite(os.path.join(opts.det, 'image', str(ep) + ".png"), img[0, :, :, :])
        '''

        # Save the training model
        if (ep % 100 == 0):
            torch.save(model.state_dict(), './Model/%d.pkl' % (ep))


if __name__ == '__main__':
    opts = TrainOptions().parse()
    train(opts)

# def test(opts):
