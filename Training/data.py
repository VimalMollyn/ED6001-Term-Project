import torch
from torch.utils.data import Dataset
from tqdm.notebook import tqdm_notebook
import numpy as np

class MRIDataset(Dataset):
    def __init__(self, path_to_data, split="train"):
        if split == "train":
            split_indices = [*range(0, 101)]
        elif split == "test":
            split_indices = [*range(100, 111)]

        print(f"Start reading {split}ing dataset...")

        # read the first set of volumes
        clean_mri_set = []
        noisy_mri_set = []

        for i in tqdm_notebook(split_indices):
            # load the current volumes
            clean_mri_temp = np.load(path_to_data / f"data/{i}.npy")
            noisy_mri_temp = np.load(path_to_data / f"noisy/{i}.npy")

            # append to the existing stack
            clean_mri_set.append(clean_mri_temp)
            noised_mri_set.append(noisy_mri_temp)

        self.clean_mri_set = np.concatenate(clean_mri_set, axis=0)
        self.noisy_mri_set = np.concatenate(noisy_mri_set, axis=0)
        self.total = self.clean_mri_set.shape[0]
        self.current_patch = 1

        print(self.clean_mri_set.shape)
        print(f"End reading {split}ing dataset...")

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        clean_img = torch.from_numpy(self.clean_mri_set[index]).float()
        noisy_img = torch.from_numpy(self.noisy_mri_set[index]).float()
        return {"clean_img": clean_img, "noisy_img": noisy_img}
