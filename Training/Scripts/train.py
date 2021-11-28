import sys
sys.path.insert(0,'../../Training/')

## imports go here
import os
import time
import tqdm
# import tqdm.notebook as tqdm

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import numpy as np
import nibabel as nib
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, normalized_root_mse

import wandb

from preprocessing import patch_test_img, merge_test_img
# from data import MRIDataset ## imports for the MRI dataset
from pathlib import Path
from collections import defaultdict
import torchvision
import torchvision.transforms.functional as F

import matplotlib.pyplot as plt

# wandb stuff
wandb.login()
wandb.init(project="mia_final")

class GeneratorNet(nn.Module):
    """ 
    In the original implementation, post-activation values were being added to pre-activation values 
    Modified it to add pre-activation values, then apply activation
    Removed final ReLU
    """

    def __init__(self):
        super(GeneratorNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(1, 32, kernel_size=3, padding=1, bias=True), nn.BatchNorm3d(32),)
        self.conv2 = nn.Sequential(
            nn.LeakyReLU(inplace=True), nn.Conv3d(32, 64, kernel_size=3, padding=1, bias=True), nn.BatchNorm3d(64),
        )
        self.conv3 = nn.Sequential(
            nn.LeakyReLU(inplace=True), nn.Conv3d(64, 128, kernel_size=3, padding=1, bias=True), nn.BatchNorm3d(128),
        )
        self.conv4 = nn.Sequential(
            nn.LeakyReLU(inplace=True), nn.Conv3d(128, 256, kernel_size=3, padding=1, bias=True), nn.BatchNorm3d(256),
        )

        self.deConv1 = nn.Sequential(
            nn.LeakyReLU(inplace=True), nn.Conv3d(256, 128, kernel_size=3, padding=1, bias=True), nn.BatchNorm3d(128),
        )

        self.deConv2 = nn.Sequential(
            nn.LeakyReLU(inplace=True), nn.Conv3d(128, 64, kernel_size=3, padding=1, bias=True), nn.BatchNorm3d(64),
        )

        self.deConv3 = nn.Sequential(
            nn.LeakyReLU(inplace=True), nn.Conv3d(64, 32, kernel_size=3, padding=1, bias=True), nn.BatchNorm3d(32),
        )

        self.deConv4 = nn.Sequential(nn.LeakyReLU(inplace=True), nn.Conv3d(32, 1, kernel_size=3, padding=1),)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        out = self.deConv1(conv4)
        out += conv3

        out = self.deConv2(out)
        out += conv2

        out = self.deConv3(out)
        out += conv1

        out = self.deConv4(out)
        out += x

        return out


class DiscriminatorNet(nn.Module):
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(1, 32, kernel_size=3, padding=1), nn.LeakyReLU(),)

        self.conv2 = nn.Sequential(nn.Conv3d(32, 64, kernel_size=3, padding=1), nn.LeakyReLU(),)

        self.conv3 = nn.Sequential(nn.Conv3d(64, 128, kernel_size=3, padding=1), nn.LeakyReLU(),)

        self.fc = nn.Linear(128 * 6 * 32 * 32, 1)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, self.num_flat_features(x))
        output = self.fc(x)

        return output

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for i in size:
            num_features *= i

        return num_features


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        self.feature = vgg19.features
    '''
    input: N*1*D(6)*H*W
    output: N*C*H*W
    '''

    def forward(self, input):
        # VGG19: means:103.939, 116.779, 123.68
        input /= 16
        depth = input.size()[2]
        result = []
        for i in range(depth):
            x = torch.cat(
                (input[:, :, i, :, :] - 103.939, input[:, :, i, :, :] - 116.779, input[:, :, i, :, :] - 123.68), 1)
            result.append(self.feature(x))

        output = torch.cat(result, dim=1)

        return output

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv3d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm3d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
                
def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

class WGAN():
    def __init__(self, level=0):
        # parameters
        self.epochs = 10
        self.batch_size = 64
        self.lr =5e-6

        self.d_iter = 5
        self.lambda_gp = 10

        self.lambda_vgg = 1e-1
        self.lambda_d = 1e-3
        self.lambda_mse = 1

        self.level = level

        self.loss_dir = "./loss/"
        self.v = "0_0_5_%d" % self.level  # vs
        self.save_dir = "./model/" + self.v + "/"
        Path(self.loss_dir).mkdir(parents=True, exist_ok=True)
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        self.gpu = False

        self.generator = GeneratorNet()
        self.discriminator = DiscriminatorNet()
        self.vgg19 = VGG19()

        self.G_optimizer = optim.Adam(
            self.generator.parameters(), lr=self.lr, betas=(0.5, 0.9))
        self.D_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.9))

        self.G_loss = nn.MSELoss()

        if torch.cuda.is_available():
            self.generator.cuda()
            self.discriminator.cuda()
            self.vgg19.cuda()
            self.gpu = True
        if not self.load_model():
            initialize_weights(self.generator)
            initialize_weights(self.discriminator)

    def train(self, trainloader, validloader):
        self.dataloader = trainloader
        self.validDataloader = validloader
        examples_seen = 0

        ## training loop
        for epoch in range(0, self.epochs):

            # iterate over the dataset
            pbar = tqdm.tqdm(total=len(self.dataloader))
            pbar.set_description(f"Epoch {epoch}")
            
            for batch_index, batch in enumerate(self.dataloader):
                clean_img = batch["clean_img"]
                noised_img = batch["noisy_img"]
                
                examples_seen += clean_img.shape[0]

                # train discriminator
                for iter_i in range(self.d_iter):
                    loss = self._train_discriminator(clean_img, noised_img)

                wandb.log({
                    "discriminator_loss": loss[0],
                    "neg_real_validity": loss[1],
                    "fake_validity": loss[2],
                    "gradient_penalty": loss[3],
                    "examples_seen": examples_seen
                })

                # train generator
                loss = self._train_generator(clean_img, noised_img)

                wandb.log({
                    "generator_loss": loss[0],
                    "mse_loss": loss[1],
                    "neg_fake_validity": loss[2],
                    "perceptual_loss": loss[3],
                    "examples_seen": examples_seen
                })


                # save model and loss
                if batch_index % 100 == 0:
                    self.save_model()
                
                # update pbar
                loss = [0]
                pbar.update(1)
                pbar.set_postfix(loss=loss[0])
                with torch.no_grad():
                    self.visualize_denoised()


            if ((epoch + 1) % 4 == 0 and self.lr > 1e-7):
                self.G_optimizer.defaults["lr"] *= 0.5
                self.G_optimizer.defaults["lr"] *= 0.5
                self.lr *= 0.5

            pbar.close()
            with torch.no_grad():
                self.test(examples_seen)


    def _train_discriminator(self, clean_img, noised_img, train=True):
        self.D_optimizer.zero_grad()

        z = Variable(noised_img)
        real_img = Variable(clean_img / 4096)
        if self.gpu:
            z = z.cuda()
            real_img = real_img.cuda()

        fake_img = self.generator(z)
        real_validity = self.discriminator(real_img)
        fake_validity = self.discriminator(fake_img.data / 4096)
        gradient_penalty = self._calc_gradient_penalty(
            real_img.data, fake_img.data)

        d_loss = torch.mean(-real_validity) + torch.mean(fake_validity) + \
            self.lambda_gp * gradient_penalty
        if train:
            d_loss.backward()
            self.D_optimizer.step()

        return d_loss.data.item(), torch.mean(-real_validity).cpu().item(), torch.mean(fake_validity).cpu().item(), self.lambda_gp * gradient_penalty.cpu().item()

    def _train_generator(self, clean_img, noised_img, train=True):
        z = Variable(noised_img)
        real_img = Variable(clean_img, requires_grad=False)


        if self.gpu:
            z = z.cuda()
            real_img = real_img.cuda()

        self.G_optimizer.zero_grad()
        self.D_optimizer.zero_grad()
        self.vgg19.zero_grad()

        criterion_mse = nn.MSELoss()
        criterion_vgg= nn.MSELoss()

        fake_img = self.generator(z)
        mse_loss = criterion_mse(fake_img, real_img)
        if train:
            (self.lambda_mse * mse_loss).backward(retain_graph=True)


        feature_fake_vgg = self.vgg19(fake_img)
        feature_real_vgg = Variable(self.vgg19(real_img).data, requires_grad=False).cuda()

        vgg_loss = criterion_vgg(feature_fake_vgg, feature_real_vgg)

        fake_validity = self.discriminator(fake_img / 4096)
        g_loss =  self.lambda_vgg * vgg_loss + self.lambda_d * torch.mean(-fake_validity)

        if train:
            g_loss.backward()
            self.G_optimizer.step()
        return g_loss.data.item(), mse_loss.data.item(), torch.mean(-fake_validity).data.item(), vgg_loss.data.item()

    def _calc_gradient_penalty(self, clean_img, gen_img):
        batch_size = clean_img.size()[0]
        alpha = Variable(torch.rand(batch_size, 1))
        alpha = alpha.expand(batch_size, clean_img.nelement(
        ) // batch_size).contiguous().view(clean_img.size()).float()
        if self.gpu:
            alpha = alpha.cuda()

        interpolates = (alpha * clean_img + (1 - alpha)
                        * gen_img).requires_grad_(True)
        disc_interpolates = self.discriminator(interpolates)
        fake = Variable(torch.Tensor(batch_size, 1).fill_(1.0),
                        requires_grad=False)
        if self.gpu:
            fake = fake.cuda()

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def test(self, examples_seen):
        self.generator.eval()
        self.discriminator.eval()
        
        timestr = time.strftime("%H:%M:%S", time.localtime())
        # print(timestr)

        # test set
        total_mse_loss = 0
        total_g_loss = 0
        total_d_loss = 0
        total_vgg_loss = 0
        batch_num = 0
        for batch_index, batch in enumerate(self.validDataloader):
            clean_img = batch["clean_img"]
            noisy_img = batch["noisy_img"]

            loss = self._train_generator(clean_img, noisy_img, train=False)
            total_g_loss += loss[0]
            total_mse_loss += loss[1]
            total_d_loss += loss[2]
            total_vgg_loss += loss[3]
            batch_num += 1
        mse_loss = total_mse_loss / batch_num
        g_loss = total_g_loss / batch_num
        d_loss = total_d_loss / batch_num
        vgg_loss = total_vgg_loss / batch_num

        wandb.log({
            "test_discriminator_loss": d_loss,
            "test_generator_loss": g_loss,
            "test_perceptual_loss": vgg_loss,
            "test_mse_loss": mse_loss,
            "examples_seen": examples_seen
            }
        )
        
        # self.compute_quality()
        self.save_loss((vgg_loss, mse_loss, g_loss, d_loss))
        self.save_model()

        self.generator.train()
        self.discriminator.train()

    def visualize_denoised(self):
        denoised = defaultdict(list)
        
        for batch_index, batch in enumerate(self.validDataloader):
            clean_img = batch["clean_img"]
            noised_img = batch["noisy_img"]
            
            if self.gpu:
                noised_img = noised_img.cuda()
            
            with torch.no_grad():
                denoised_img = self.generator(noised_img)
                denoised["clean"].append(clean_img[:, :, 3])
                denoised["denoised"].append(denoised_img[:, :, 3])
                denoised["noisy"].append(noised_img[:, :, 3])
            break
            
        
        # denoised is a list of denoised patches
        # now take a sample slice from each patch and plot
        clean_img_grid = torchvision.utils.make_grid(denoised["clean"][0])
        noisy_img_grid = torchvision.utils.make_grid(denoised["noisy"][0])
        denoised_img_grid = torchvision.utils.make_grid(denoised["denoised"][0])
        
        # show(clean_img_grid)
        # show(noisy_img_grid)
        # show(denoised_img_grid)
        
        wandb.log({
            "Clean Images": wandb.Image(clean_img_grid),
            "Noisy Images": wandb.Image(noisy_img_grid),
            "Denoised Images": wandb.Image(denoised_img_grid)
        })
        
    def compute_quality(self):
        psnr1 = 0
        psnr2 = 0
        mse1 = 0
        mse2 = 0
        ssim1 = 0
        ssim2 = 0
        _psnr1 = 0
        _psnr2 = 0
        _mse1 = 0
        _mse2 = 0
        _ssim1 = 0
        _ssim2 = 0

        for i in range(101, 111):
            # read a clean image and it's noisy counterpart,
            # convert noisy image into it's patches and obtain a cleaned image
            # merge the patches together
            # compute metrics
            
            clean_mri = np.load(path_to_data / f"data/{i}.npy").squeeze(axis=1) ## shape (784, 1, 6, 32, 32)
            noisy_mri = np.load(path_to_data / f"noisy/{i}.npy").squeeze(axis=1)
            
            # pass the noisy_mri through the generator
            with torch.no_grad():
                denoised = self.generator(noisy_mri)
                
            
#             patchs, row, col = patch_test_img(noisy_img)
#             denoisy_img = merge_test_img(
#                 self.denoising(patchs), row, col).astype(np.int16)

            psnr1 += peak_signal_noise_ratio(clean_img, noisy_img, 4096)
            psnr2 += peak_signal_noise_ratio(clean_img, denoisy_img, 4096)

            mse1 += normalized_root_ms(clean_img, noisy_img)
            mse2 += normalized_root_ms(clean_img, denoisy_img)

            ssim1 += structural_similarity(clean_img, noisy_img,
                                  data_range=4096, multichannel=True)
            ssim2 += structural_similarity(clean_img, denoisy_img,
                                  data_range=4096, multichannel=True)

            max = np.max(clean_img)
            _psnr1 += peak_signal_noise_ratio(clean_img, noisy_img, max)
            _psnr2 += peak_signal_noise_ratio(clean_img, denoisy_img, max)

            _mse1 += normalized_root_ms(clean_img, noisy_img)
            _mse2 += normalized_root_ms(clean_img, denoisy_img)

            _ssim1 += structural_similarity(clean_img, noisy_img,
                                   data_range=max, multichannel=True)
            _ssim2 += structural_similarity(clean_img, denoisy_img,
                                   data_range=max, multichannel=True)
            
        psnr1 *= 0.1
        psnr2 *= 0.1
        mse1 *= 0.1
        mse2 *= 0.1
        ssim1 *= 0.1
        ssim2 *= 0.1

        _psnr1 *= 0.1
        _psnr2 *= 0.1
        _mse1 *= 0.1
        _mse2 *= 0.1
        _ssim1 *= 0.1
        _ssim2 *= 0.1
        with open("./loss/" + self.v + "psnr.csv", "a+") as f:
            f.write("%f,%f,%f,%f,%f,%f\n" %
                    (_psnr1, _psnr2, _ssim1, _ssim2, _mse1, _mse2))
        timestr = time.strftime("%H:%M:%S", time.localtime())
        with open("./loss/" + self.v + "psnr_4096.csv", "a+") as f:
            f.write("%s: %.10f,%f,%f,%f,%f,%f,%f\n" %
                    (timestr, self.lr, psnr1, psnr2, ssim1, ssim2, mse1, mse2))
        print("psnr: %f,%f,ssim: %f,%f,mse:%f,%f\n" %
              (_psnr1, _psnr2, _ssim1, _ssim2, _mse1, _mse2))

    '''
    N*H*W*D -> N*C*D*H*W
    return: N*H*W*D
    '''

#     def denoising(self, patchs):
#         n, h, w, d = patchs.shape
#         denoised_patchs = []
#         for i in range(0, n, self.batch_size):
#             batch = patchs[i:i + self.batch_size]
#             batch_size = batch.shape[0]
#             x = np.reshape(batch, (batch_size, 1, w, h, d))
#             x = x.transpose(0, 1, 4, 2, 3)
#             x = Variable(torch.from_numpy(x).float()).cuda()
#             y = self.generator(x)
#             denoised_patchs.append(y.cpu().data.numpy())

#         denoised_patchs = np.vstack(denoised_patchs)
#         denoised_patchs = np.reshape(denoised_patchs, (n, d, h, w))
#         denoised_patchs = denoised_patchs.transpose(0, 2, 3, 1)
#         return denoised_patchs

    def save_model(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        torch.save(self.generator.state_dict(),
                   self.save_dir + "G_" + self.v + ".pkl")
        torch.save(self.discriminator.state_dict(),
                   self.save_dir + "D_" + self.v + ".pkl")

    def load_model(self):
        if os.path.exists(self.save_dir + "G_" + self.v + ".pkl") and \
                os.path.exists(self.save_dir + "D_" + self.v + ".pkl"):

            self.generator.load_state_dict(
                torch.load(self.save_dir + "G_" + self.v + ".pkl")
            )
            self.discriminator.load_state_dict(
                torch.load(self.save_dir + "D_" + self.v + ".pkl")
            )
            return True
        else:
            return False

    def save_loss(self, loss):
        value = ""
        for item in loss:
            value = value + str(item) + ","
        value += "\n"
        with open("./loss/" + self.v + ".csv", "a+") as f:
            f.write(value)

import torch
from torch.utils.data import Dataset
from tqdm.notebook import tqdm_notebook
import numpy as np

class MRIDataset(Dataset):
    def __init__(self, path_to_data, split="train"):
        if split == "train":
            split_indices = [*range(1, 101)]
        elif split == "test":
            split_indices = [*range(101, 111)]

        print(f"Start reading {split}ing dataset...")

        # read the first set of volumes
        clean_mri_set = []
        noisy_mri_set = []

        for i in tqdm_notebook(split_indices):
            # load the current volumes
            clean_mri_temp = np.load(path_to_data / f"data/{i}.npy").squeeze(axis=1)
            noisy_mri_temp = np.load(path_to_data / f"noisy/{i}.npy").squeeze(axis=1)

            # append to the existing stack
            clean_mri_set.append(clean_mri_temp)
            noisy_mri_set.append(noisy_mri_temp)

        self.clean_mri_set = np.concatenate(clean_mri_set, axis=0)
        self.noisy_mri_set = np.concatenate(noisy_mri_set, axis=0)
        # self.arrshape0 = clean_mri_set[0].shape[0]
        # self.total = len(self.clean_mri_set) * self.arrshape0
        self.total = self.clean_mri_set.shape[0]
        self.current_patch = 1

        print(len(self.clean_mri_set))
        print(self.total)
        print(f"End reading {split}ing dataset...")

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        # x = index // self.arrshape0
        # y = index % self.arrshape0
        # clean_img = torch.from_numpy(self.clean_mri_set[x][y]).float().squeeze(axis=1)
        # noisy_img = torch.from_numpy(self.noisy_mri_set[x][y]).float().squeeze(axis=1)
        
        clean_img = torch.from_numpy(self.clean_mri_set[index]).float()
        noisy_img = torch.from_numpy(self.noisy_mri_set[index]).float()
        return {"clean_img": clean_img, "noisy_img": noisy_img}

path_to_data = Path("../../patches/")

trainset = MRIDataset(path_to_data=path_to_data, split="train")
validset = MRIDataset(path_to_data=path_to_data, split="test")

wgan = WGAN()
trainloader = DataLoader(trainset, batch_size=wgan.batch_size, shuffle=True)
validloader = DataLoader(validset, batch_size=wgan.batch_size, shuffle=True)
wgan.train(trainloader, validloader)
