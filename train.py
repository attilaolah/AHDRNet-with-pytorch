"""Script to train the model.

Author: Wei Wang
"""

import os

import numpy as np
import torch
from torch import autograd
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from datsetprocess import AHDRDataset
from model import AHDRNet
from opts import TrainOptions
from utils import batch_PSNR

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def train(opts, learn_rate: int = 0.0001) -> None:
    # Create the loader:
    train_data = AHDRDataset(scene_directory=opts.folder)
    loader = DataLoader(train_data, batch_size=opts.batch_size,
                        shuffle=True, num_workers=1)
    # Create the model:
    model = AHDRNet().cuda()
    criterion = nn.L1Loss().to(opts.device)
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    loss_list = []
    current_epoch = 0
    # Load pre-train model:
    if os.path.exists(opts.resume):
        resume_file = os.path.realpath(opts.resume)
        basename, _ = os.path.splitext(os.path.basename(resume_file))
        current_epoch = int(basename) + 1
        state = torch.load(opts.resume)
        loss_list = state['loss_list']
        model.load_state_dict(state['model'])

    # Train:
    progress_bar = tqdm(range(current_epoch, opts.epoch), unit='epoch',
                        initial=current_epoch)
    progress_bar.set_description('[ AHDRNet ]')
    for epoch in progress_bar:
        losses = []
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

            # Forward and compute loss:
            pre = model(batch_x1, batch_x2, batch_x3)
            loss = criterion(pre, batch_x4)
            psnr = batch_PSNR(torch.clamp(pre, 0., 1.), batch_x4, 1.0)
            losses.append(loss.item())
            progress_bar.set_description(
                '[ AHDRNet ] STEP {}/{} | LOSS {:0.6f} | PSNR {:0.6f}'
                .format(step+1, len(loader), losses[-1], psnr))

            # Update the parameters:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_list.append(np.mean(losses))

        # Save the training model.
        if epoch > 0 and epoch % opts.record_epoch == 0:
            # Save progress:
            save_dir = os.path.join(opts.det, 'model')
            save_file = '{:06d}.pkl'.format(epoch)
            save_path = os.path.join(save_dir, save_file)
            torch.save({
                'model': model.state_dict(),
                'loss_list': loss_list,
            }, save_path)

            # Create a symlink for easy resume:
            latest = os.path.join(save_dir, 'latest.pkl')
            if os.path.exists(latest):
                os.unlink(latest)
            os.symlink(save_file, latest)


if __name__ == '__main__':
    train(TrainOptions().parse())
