"""Create Saliency Maps."""
import os
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader

from common import FIGURES_DIR
from utils import load_dataset, load_model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Arguments
def parse_args():
    """Parse script arguments.

    Returns:
        Namespace with model name, checkpoints path and dataset name.
    """
    parser = argparse.ArgumentParser(description='Plot saliency maps.')
    parser.add_argument('--model', '-m',
                        default='XceptionBased', type=str,
                        help='Model name: SimpleNet or XceptionBased.')
    parser.add_argument('--checkpoint_path', '-cpp',
                        default='checkpoints/XceptionBased.pt', type=str,
                        help='Path to model checkpoint.')
    parser.add_argument('--dataset', '-d',
                        default='fakes_dataset', type=str,
                        help='Dataset: fakes_dataset or synthetic_dataset.')

    return parser.parse_args()


def compute_gradient_saliency_maps(samples: torch.tensor,
                                   true_labels: torch.tensor,
                                   model: nn.Module):
    """Compute vanilla gradient saliency maps for the samples.

    Recipe:
        (1) Set requires_grad_ for the samples.
        (2) Compute a forward pass for the samples in the model.
        (3) Gather only the scores which corresponds to the true labels of the
        samples.
        (4) Compute a backward pass on these scores.
        (5) Collect the gradients from the samples object.
        (6) Compute the absolute value (L1) of these values.
        (7) Pick the maximum value from channels on each pixel.

    Args:
        samples: The samples we want to compute saliency maps for. Tensor of
        shape Bx3x256x256.
        true_labels: The true labels of the samples. Tensor of shape (B,).
        model: The model we want to compute the gradients for.
    Returns:
        saliency: vanilla gradient saliency maps. This should be a tensor of
        shape Bx256x256 where B is the number of images in samples.
    """
    """INSERT YOUR CODE HERE, overrun return."""
    return torch.rand(6, 256, 256)


def main():  # pylint: disable=R0914, R0915
    """Parse script arguments, show saliency maps for 36 random samples,
    and the average saliency maps over all real and fake samples in the test
    set."""
    args = parse_args()

    # load dataset
    test_dataset = load_dataset(dataset_name=args.dataset, dataset_part='test')

    # load model
    model_name = args.model
    model = load_model(model_name)
    model.load_state_dict(torch.load(args.checkpoint_path)['model'])
    model.eval()

    # create sets of samples of images and their corresponding saliency maps
    all_samples = []
    all_saliency_maps = []
    sample_to_image = lambda x: np.transpose(x, (1, 2, 0))

    for _ in range(6):
        samples, true_labels = next(iter(DataLoader(test_dataset,
                                                    batch_size=6,
                                                    shuffle=True)))
        all_samples.append(torch.cat([sample_to_image(s).unsqueeze(0)
                                      for s in samples]))
        saliency_maps = compute_gradient_saliency_maps(samples.to(device),
                                     true_labels.to(device),
                                     model)
        all_saliency_maps.append(saliency_maps.cpu().detach())

    all_samples = torch.cat(all_samples)
    all_saliency_maps = torch.cat(all_saliency_maps)

    saliency_maps_and_images_pairs = plt.figure()
    plt.suptitle('Images and their saliency maps')
    for idx, (image, saliency_map) in enumerate(zip(all_samples,
                                                    all_saliency_maps)):
        plt.subplot(6, 6 * 2, 2 * idx + 1)
        # plot image
        image -= image.min()
        image /= image.max()
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        # plot saliency map
        plt.subplot(6, 6 * 2, 2 * idx + 2)
        saliency_map -= saliency_map.min()
        saliency_map /= saliency_map.max()
        plt.imshow(saliency_map)
        plt.xticks([])
        plt.yticks([])

    saliency_maps_and_images_pairs.set_size_inches((8, 8))
    saliency_maps_and_images_pairs.savefig(
        os.path.join(FIGURES_DIR,
                     f'{args.dataset}_{args.model}_'
                     f'saliency_maps_and_images_pairs.png'))

    # loop through the images in the test set and compute saliency map for
    # each image. Compute the average map of all real face image and
    # all fake face image images.
    dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    real_images_saliency_maps = []
    fake_images_saliency_maps = []

    for samples, true_labels in dataloader:
        fake_samples = samples[true_labels == 1].to(device)
        fake_labels = true_labels[true_labels == 1].to(device)
        real_samples = samples[true_labels == 0].to(device)
        real_labels = true_labels[true_labels == 0].to(device)
        saliency_maps = compute_gradient_saliency_maps(fake_samples,
                                                       fake_labels,
                                                       model)
        fake_images_saliency_maps.append(saliency_maps.cpu().detach())
        saliency_maps = compute_gradient_saliency_maps(real_samples,
                                                       real_labels,
                                                       model)
        real_images_saliency_maps.append(saliency_maps.cpu().detach())

    all_real_saliency_maps = torch.cat(real_images_saliency_maps)
    all_fake_saliency_maps = torch.cat(fake_images_saliency_maps)

    for idx in range(all_real_saliency_maps.shape[0]):
        all_real_saliency_maps[idx] -= all_real_saliency_maps[idx].min()
        all_real_saliency_maps[idx] /= all_real_saliency_maps[idx].max()

    for idx in range(all_fake_saliency_maps.shape[0]):
        all_fake_saliency_maps[idx] -= all_fake_saliency_maps[idx].min()
        all_fake_saliency_maps[idx] /= all_fake_saliency_maps[idx].max()

    mean_saliency_maps = plt.figure()
    plt.subplot(1, 2, 1)
    mean_map = all_fake_saliency_maps.mean(axis=0)
    mean_map -= mean_map.min()
    mean_map /= mean_map.max()
    plt.imshow(mean_map)
    plt.title('mean of fake images saliency maps')
    plt.subplot(1, 2, 2)
    mean_map = all_real_saliency_maps.mean(axis=0)
    mean_map -= mean_map.min()
    mean_map /= mean_map.max()
    plt.imshow(mean_map)
    plt.title('mean of real images saliency maps')
    mean_saliency_maps.set_size_inches((8, 6))
    mean_saliency_maps.savefig(
        os.path.join(FIGURES_DIR,
                     f'{args.dataset}_{args.model}_mean_saliency_maps.png'))


if __name__ == '__main__':
    main()
