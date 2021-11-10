import nibabel as nib
import numpy as np


def patch_test_img(img, size=32):
    """
    Take in a single image, generate sliding-window patches of size (size x size)
    Return the patches stacked along a new axis
    """

    patchs = []
    height, width, depth = img.shape

    row = 0

    for i in range(0, height - size + 1, size // 2):
        row += 1
        col = 0
        for j in range(0, width - size + 1, size // 2):
            col += 1
            patchs.append(img[i : i + size, j : j + size, :])
    temp = np.vstack(patchs)
    temp = np.reshape(temp, (-1, size, size, depth))
    return temp, row, col


def merge_test_img(patchs, row, col, size=32):
    """
    Take in an array of patches stacked along the first axis
    Reconstruct the image from which the patches were generated
    Return the image
    """

    patchs_num = patchs.shape[0]
    rows = []
    x = size // 8
    y = size // 4
    row_index = 0
    for i in range(0, patchs_num, col):
        temp = patchs[i, :, :-x, :]
        for j in range(1, col - 1):
            temp[:, -y:, :] = (temp[:, -y:, :] + patchs[i + j, :, x : x + y, :]) / 2
            temp = np.append(temp, patchs[i + j, :, x + y : -x, :], axis=1)
        temp[:, -y:, :] = (temp[:, -y:, :] + patchs[i + j + 1, :, x : x + y, :]) / 2
        temp = np.append(temp, patchs[i + j + 1, :, x + y :, :], axis=1)

        row_index += 1
        rows.append(temp)
    img = rows[0][:-x, :, :]
    length = len(rows)
    for i in range(1, length - 1):
        img[-y:, :, :] = (img[-y:, :, :] + rows[i][x : x + y, :, :]) / 2
        img = np.append(img, rows[i][x + y : -x, :, :], axis=0)
    img[-y:, :, :] = (img[-y:, :, :] + rows[-1][x : x + y, :, :]) / 2
    img = np.append(img, rows[-1][x + y :, :, :], axis=0)

    return img
