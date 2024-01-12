"""Utility methods and constants used throughout the project."""
import os

import torch
from torch import nn
from torchvision import transforms

from faces_dataset import FacesDataset
from models import SimpleNet, get_xception_based_model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRANSFORM_TRAIN = transforms.Compose([
    transforms.RandomCrop(256, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

TRANSFORM_TEST = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])


def load_dataset(dataset_name: str, dataset_part: str) -> \
        torch.utils.data.Dataset:
    """Loads dataset part from dataset name.

    For example, loading the trining set of the Deepfakes dataset:
    >>> deepfakes_train = load_dataset('fakes_dataset', 'train')

    Args:
        dataset_name: dataset name, one of: fakes_dataset, synthetic_dataset.
        dataset_part: dataset part, one of: train, val, test.

    Returns:
        dataset: a torch.utils.dataset.Dataset instance.
    """
    transform = {'train': TRANSFORM_TRAIN,
                 'val': TRANSFORM_TEST,
                 'test': TRANSFORM_TEST}[dataset_part]
    dataset = FacesDataset(
        root_path=os.path.join('..',
                               'Assignment4_datasets',
                               dataset_name,
                               dataset_part),
        transform=transform)
    return dataset


def load_model(model_name: str) -> nn.Module:
    """Load the model corresponding to the name given.

    Args:
        model_name: the name of the model, one of: SimpleNet, XceptionBased.

    Returns:
        model: the model initialized, and loaded to device.
    """
    models = {
        'SimpleNet': SimpleNet(),
        'XceptionBased': get_xception_based_model(),
    }

    if model_name not in models:
        raise ValueError(f"Invalid Model name {model_name}")

    print(f"Building model {model_name}...")
    model = models[model_name]
    model = model.to(device)
    return model


def get_nof_params(model: nn.Module) -> int:
    """Return the number of trainable model parameters.

    Args:
        model: nn.Module.

    Returns:
        The number of model parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
