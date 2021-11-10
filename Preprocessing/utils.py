import nibabel as nib
import numpy as np

# define a function to add rician noise
## taken from the paper
def add_rice_noise(img, snr=10, mu=0.0, sigma=1):
    level = snr * np.max(img) / 100
    size = img.shape
    x = level * np.random.normal(mu, sigma, size=size) + img
    y = level * np.random.normal(mu, sigma, size=size)
    return np.sqrt(x**2 + y**2).astype(np.int16)

def add_noise_and_save(params):
    # unpack params
    file_path = params["file_path"]
    path_to_save = params["path_to_save"]
    
    # load image and add noise
    nii_img = nib.load(file_path)
    img_data = nii_img.get_fdata()
    noisy_img = add_rice_noise(img_data)
    noisy_nii = nib.Nifti1Image(noisy_img, nii_img.affine, nii_img.header)
    nib.save(noisy_nii, path_to_save / file_path.name) # saving as nii saves some space
    return None
