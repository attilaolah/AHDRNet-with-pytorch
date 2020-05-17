"""This script defines the procedure to parse the parameters

Author: Wei Wang
"""
import argparse
import enum
import json
import os
import torch


class Device(enum.Enum):
    CPU = 'cpu'
    GPU = 'cuda'


class Options:
    PREFIX: str = '[ AHDRNet ] '

    # Device type (CPU or GPU/CUDA).
    device: Device = Device.CPU

    def __init__(self) -> None:
        if torch.cuda.is_available:
            self.device = Device.GPU

    @classmethod
    def _present_parameters(cls, namespace: argparse.Namespace) -> None:
        """Print the parameters line by line."""
        print('{}: '.format(cls.__name__))
        print(json.dumps(vars(namespace), indent=2, sort_keys=True))


class TrainingOptions(Options):
    """Training options."""

    # Maximum epoch. Training will stop after this many iterations.
    max_epoch: int = 15000

    # Path to the bracketed images.
    training_data: str = os.path.join('data', 'Training')

    # Batch size. ~20 seems to use about 7Gb of memory on the GPU.
    batch_size: int = 8

    # Record a checkpoint at every this many iterations.
    checkpoint_interval: int = 1

    # Where to store the train results.
    checkpoint_directory: str = 'model'

    # Path of the pre-trained model. If exists, resume training from here.
    checkpoint: str = os.path.join(checkpoint_directory, 'latest.pkl')

    learn_rate: float = 0.0001

    def __init__(self) -> None:
        super().__init__()
        parser = argparse.ArgumentParser()
        parser.add_argument('--max_epoch', type=int, default=self.max_epoch)
        parser.add_argument('--training_data', type=str,
                            default=self.training_data)
        parser.add_argument('--batch_size', type=int, default=self.batch_size)
        parser.add_argument('--checkpoint_interval', type=int,
                            default=self.checkpoint_interval)
        parser.add_argument('--checkpoint_directory', type=str,
                            default=self.checkpoint_directory)
        parser.add_argument('--checkpoint', type=str, default=self.checkpoint)

        args = parser.parse_args()
        self._present_parameters(args)

        for key, value in vars(args).items():
            setattr(self, key, value)

        # Create the resulting directories:
        if not os.path.exists(self.checkpoint_directory):
            os.mkdir(self.checkpoint_directory)


class TestOptions(Options):
    """Testing options."""

    # Path to the under-exposed image.
    image_1: str

    # Path to the over-exposed image.
    image_2: str

    # Path to the pre-trained model.
    model: str = os.path.join('checkpoint_directory', 'model', 'latest.pkl')

    # Path to the fused image.
    result: str = 'result.png'

    # Height and width of the resulting (fused) image.
    height: int = 400
    width: int = 600

    def __init__(self):
        super().__init__()
        parser = argparse.ArgumentParser()
        parser.add_argument('--image_1', type=str, required=True)
        parser.add_argument('--image_2', type=str, required=True)
        parser.add_argument('--model', type=str, default=self.model)
        parser.add_argument('--result', type=str, default=self.result)
        parser.add_argument('--height', type=int, default=self.height)
        parser.add_argument('--width', type=int, default=self.width)

        args = parser.parse_args()
        self._present_parameters(args)

        for key, value in vars(args).items():
            setattr(self, key, value)
