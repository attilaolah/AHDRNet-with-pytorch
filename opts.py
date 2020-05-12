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
        print(json.dumps(vars(namespace), indent=2))


class TrainingOptions(Options):
    """Training options."""

    # Path to the bracketed images.
    training_data: str = os.path.join('data', 'Training')

    # Batch size. ~20 seems to use about 6215Mb of memory on the GPU.
    batch_size: int = 8

    # Where to store the train results.
    train_results: str = 'train_results'

    # Path of the pre-trained model. If exists, resume training from here.
    checkpoint: str = os.path.join(train_results, 'model', 'latest.pkl')

    # Maximum epoch. Training will stop after this many iterations.
    max_epoch: int = 15000

    # Record a checkpoint at every this many iterations.
    checkpoint_interval: int = 1

    # Directories to store images and models.
    image_directory: str
    model_directory: str

    learn_rate: float = 0.0001

    def __init__(self) -> None:
        super().__init__()
        parser = argparse.ArgumentParser()
        parser.add_argument('--training_data', type=str,
                            default=self.training_data)
        parser.add_argument('--batch_size', type=int, default=self.batch_size)
        parser.add_argument('--train_results', type=str,
                            default=self.train_results)
        parser.add_argument('--checkpoint', type=str, default=self.checkpoint)
        parser.add_argument('--max_epoch', type=int, default=self.max_epoch)
        parser.add_argument('--checkpoint_interval', type=int,
                            default=self.checkpoint_interval)
        args = parser.parse_args()

        self._present_parameters(args)

        # Create the resulting directories:
        if not os.path.exists(args.train_results):
            os.mkdir(args.train_results)
        self.image_directory = os.path.join(args.train_results, 'image')
        if not os.path.exists(self.image_directory):
            os.mkdir(self.image_directory)
        self.model_directory = os.path.join(args.train_results, 'model')
        if not os.path.exists(self.model_directory):
            os.mkdir(self.model_directory)

        for key, value in vars(args).items():
            setattr(self, key, value)


class TestOptions(Options):
    """Testing options."""

    # Path to the under-exposed image.
    image_1: str

    # Path to the over-exposed image.
    image_2: str

    # Path to the pre-trained model.
    model: str = os.path.join('train_results', 'model', 'latest.pkl')

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
