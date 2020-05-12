"""Script to train the model.

Author: Wei Wang
"""

import os

import numpy as np
import torch
from torch import autograd
from torch import nn
from torch import optim
from torch.utils import data
import tqdm

import datsetprocess
import model
import opts
import utils

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def train(args: opts.TrainingOptions) -> None:
    # Create the loader:
    training_data = datsetprocess.AHDRDataset(
        scene_directory=args.training_data)
    loader = data.DataLoader(training_data, batch_size=args.batch_size,
                             shuffle=True, num_workers=1)
    # Create the model:
    ahdr_model = model.AHDRNet().cuda()
    criterion = nn.L1Loss().to(args.device.value)
    optimizer = optim.Adam(ahdr_model.parameters(), lr=args.learn_rate)

    loss_list = []
    current_epoch = 0
    # Load pre-train model:
    if os.path.exists(args.checkpoint):
        checkpoint_file = os.path.realpath(args.checkpoint)
        basename, _ = os.path.splitext(os.path.basename(checkpoint_file))
        current_epoch = int(basename) + 1
        state = torch.load(args.checkpoint)
        loss_list = state['loss_list']
        ahdr_model.load_state_dict(state['model'])

    # Train:
    progress_bar = tqdm.tqdm(range(current_epoch, args.max_epoch),
                             unit='epoch', initial=current_epoch)
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
            pre = ahdr_model(batch_x1, batch_x2, batch_x3)
            loss = criterion(pre, batch_x4)
            psnr = utils.batch_psnr(torch.clamp(pre, 0., 1.), batch_x4, 1.0)
            losses.append(loss.item())
            progress_bar.set_description(
                'STEP {}/{} | LOSS {:0.6f} | PSNR {:0.6f}'
                .format(step+1, len(loader), losses[-1], psnr))

            # Update the parameters:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_list.append(np.mean(losses))

        # Save the training model.
        if epoch > 0 and epoch % args.checkpoint_interval == 0:
            # Save progress:
            save_file = '{:06d}.pkl'.format(epoch)
            save_path = os.path.join(args.model_directory, save_file)
            torch.save({
                'model': ahdr_model.state_dict(),
                'loss_list': loss_list,
            }, save_path)

            # Create a symlink for easy resume:
            latest = os.path.join(args.model_directory, 'latest.pkl')
            if os.path.exists(latest):
                os.unlink(latest)
            os.symlink(save_file, latest)


if __name__ == '__main__':
    train(opts.TrainingOptions())
