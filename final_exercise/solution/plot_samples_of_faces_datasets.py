"""Plot samples from the Deepfakes and Synthetic Faces datasets."""
import os
import random
import matplotlib.pyplot as plt

from common import FIGURES_DIR
from utils import load_dataset


def normalize(image):
    """Normalize an image pixel values to [0, ..., 1]."""
    return (image - image.min()) / (image.max() - image.min())


def main():
    """Load the Deepfakes and Synthetic Faces datasets, sample real and fake
    images from them and plot them in a single image."""
    # create deepfakes dataset
    fakes_dataset_train = load_dataset('fakes_dataset', 'train')
    # sample indices of real and fake images
    real_image_idx = random.choice(range(int(len(fakes_dataset_train) / 2)))
    fake_image_idx = random.choice(range(int(len(fakes_dataset_train) / 2),
                                         len(fakes_dataset_train)))
    images_samples = plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(normalize(
        fakes_dataset_train[real_image_idx][0]).permute(1, 2, 0))
    plt.title('Deepfakes dataset real image')
    plt.subplot(2, 2, 2)
    plt.imshow(normalize(
        fakes_dataset_train[fake_image_idx][0]).permute(1, 2, 0))
    plt.title('Deepfakes dataset fake image')

    # create synthetic faces dataset
    synthetic_dataset_train = load_dataset('synthetic_dataset', 'train')

    real_image_idx = random.choice(range(int(len(synthetic_dataset_train) / 2)))
    fake_image_idx = random.choice(range(int(len(synthetic_dataset_train) / 2),
                                         len(synthetic_dataset_train)))
    plt.subplot(2, 2, 3)
    plt.imshow(normalize(
        synthetic_dataset_train[real_image_idx][0]).permute(1, 2, 0))
    plt.title('Synthetic images dataset real image')
    plt.subplot(2, 2, 4)
    plt.imshow(normalize(
        synthetic_dataset_train[fake_image_idx][0]).permute(1, 2, 0))
    plt.title('Synthetic images dataset fake image')
    images_samples.set_size_inches((8, 8))
    images_samples.savefig(os.path.join(FIGURES_DIR, 'datasets_samples.png'))


if __name__ == "__main__":
    main()
