"""
Run this once to generate the processed dataset ready for training
Assumes data is in a directory named 'data' located one level up
"""

import os
import nibabel as nib
import numpy as np


def add_rice_noise(img, snr=10, mu=0.0, sigma=1):
    level = snr * np.max(img) / 100
    size = img.shape
    x = level * np.random.normal(mu, sigma, size=size) + img
    y = level * np.random.normal(mu, sigma, size=size)
    return np.sqrt(x ** 2 + y ** 2).astype(np.int16)


def generate_noised_mri(noisy_level):
    """
    Load a set of noise-free images, apply noise and save it back
    """

    files = os.listdir("../data/dataset/Free/")
    for file in files:
        nii_img = nib.load("../data/dataset/Free/" + file)
        free_image = nii_img.get_data()
        noised_image = add_rice_noise(free_image, snr=noisy_level)
        noised_image = nib.Nifti1Image(noised_image, nii_img.affine, nii_img.header)
        if not os.path.exists("../data/dataset/noise_%d/" % noisy_level):
            os.makedirs("../data/dataset/noise_%d/" % noisy_level)
        nib.save(noised_image, "../data/dataset/noise_%d/%s" % (noisy_level, file))


def generate_patch(level):
    """
    Load a set of noise-free and noisy images
    Generate sliding-window patches
    Save the patches back to disk for use in the training process
    """

    stride = 8
    size = 32
    depth = 6
    num = 0
    files = os.listdir("../data/dataset/Free/")

    for file in files:
        free_img = nib.load("../data//dataset/Free/" + file).get_data()
        noised_img = nib.load("../data/dataset/noise_%d/%s" % (level, file)).get_data()
        free_img_set = None
        noised_img_set = None
        height, width, _ = free_img.shape
        for y in range(0, height - size, stride):
            for x in range(0, width - size, stride):
                free_img_temp = free_img[y : y + size, x : x + size].copy().transpose(2, 0, 1)
                noised_img_temp = noised_img[y : y + size, x : x + size].copy().transpose(2, 0, 1)
                free_img_temp = np.reshape(free_img_temp, (1, 1, depth, size, size))
                noised_img_temp = np.reshape(noised_img_temp, (1, 1, depth, size, size))

                if free_img_set is None:
                    free_img_set = free_img_temp
                    noised_img_set = noised_img_temp
                else:
                    free_img_set = np.append(free_img_set, free_img_temp, axis=0)
                    noised_img_set = np.append(noised_img_set, noised_img_temp, axis=0)
        num += 1
        print("-------" + str(num) + "-----------")
        print(noised_img_set.shape)
        print(free_img_set.shape)
        if not os.path.exists("../data/patchs32_32_%d/free/" % level):
            os.makedirs("../data/patchs32_32_%d/free/" % level)
            os.makedirs("../data/patchs32_32_%d/noised/" % level)
        np.save("../data/patchs32_32_%d/free/%d.npy" % (level, num), free_img_set)
        np.save("../data/patchs32_32_%d/noised/%d.npy" % (level, num), noised_img_set)


if __name__ == "__main__":
    pass
    # generate_noised_mri(15)
    # generate_patch(15)
