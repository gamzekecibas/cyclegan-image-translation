### CYCLEGAN MODEL TRAINING WITH PYTORCH ###
### USING AFHQ DATASET BY DEFAULT ###
## TRAINING ON CPU BY DEFAULT ###
## TRAINING ON GPU IF AVAILABLE ###

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import guided_diffusion as diffusion
import wandb

wandb.init(project="DiffGAN", entity="gkecibas16")

class DiffGAN(nn.Module):
    ## The class must be used diffusion/class GaussianDiffusion
    def __init__(self, diffusion, model, device):
        super().__init__()
        self.diffusion = diffusion
        self.model = model
        self.device = device

    def forward(self, x, t, num_samples=1):
        # x: input image
        # t: time step
        # num_samples: number of samples to generate
        # returns: a list of images
        x = x.to(self.device)
        images = []
        for i in range(num_samples):
            x = self.diffusion(x, t)
            x = self.model(x)
            images.append(x)
        return images

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 512, 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, 4, 2, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(512)
        self.conv7 = nn.Conv2d(512, 512, 4, 2, 1, bias=False)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 1, 4, 1, 0, bias=False)

    def forward(self, x):
        x1 = F.leaky_relu(self.conv1(x), 0.2)
        x2 = F.leaky_relu(self.bn2(self.conv2(x1)), 0.2)
        x3 = F.leaky_relu(self.bn3(self.conv3(x2)), 0.2)
        x4 = F.leaky_relu(self.bn4(self.conv4(x3)), 0.2)
        x5 = F.leaky_relu(self.bn5(self.conv5(x4)), 0.2)
        x6 = F.leaky_relu(self.bn6(self.conv6(x5)), 0.2)
        x7 = F.leaky_relu(self.bn7(self.conv7(x6)), 0.2)
        x8 = self.conv8(x7)
        return x8

class GaussianDiffusion(nn.Module):
    def __init__(self, sigma, device):
        super().__init__()
        self.sigma = sigma
        self.device = device

    def forward(self, x, t):
        # x: input image
        # t: time step
        # returns: a new image
        x = x.to(self.device)
        x = x + torch.randn(x.size()).to(self.device) * self.sigma * np.sqrt(t)
        return x

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv3 = nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.bn7 = nn.BatchNorm2d(64)
        self.conv8 = nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x4 = F.relu(self.bn4(self.conv4(x3)))
        x5 = F.relu(self.bn5(self.conv5(x4)))
        x6 = F.relu(self.bn6(self.conv6(x5)))
        x7 = F.relu(self.bn7(self.conv7(x6)))
        x8 = self.tanh(self.conv8(x7))
        return x8

class CGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()

    def forward(self, x):
        return self.generator(x)

def train(model, optimizer, dataloader, device, diffusion, t):
    # Train model
    for epoch in range(200):
        for i, data in enumerate(dataloader, 0):
            # Get data
            real_image, _ = data
            real_image = real_image.to(device)
            # Generate noise
            noise = torch.randn(real_image.size(0), 100, 1, 1).to(device)
            # Generate fake image
            fake_image = model.generator(noise)
            # Diffuse fake image
            fake_image = diffusion(fake_image, t)
            # Train discriminator
            optimizer.zero_grad()
            d_real = model.discriminator(real_image)
            d_fake = model.discriminator(fake_image)
            d_loss = torch.mean(d_fake) - torch.mean(d_real)
            d_loss.backward(retain_graph=True)
            optimizer.step()
            # Train generator
            optimizer.zero_grad()
            g_loss = -torch.mean(d_fake)
            g_loss.backward(retain_graph=True)
            optimizer.step()
            # Log losses
            wandb.log({"d_loss": d_loss, "g_loss": g_loss})
            # Print losses
            print('Epoch: {}, Batch: {}, d_loss: {}, g_loss: {}'.format(epoch, i, d_loss.item(), g_loss.item()))

def main():
    # Set up wandb
    wandb.init(project="cgan")
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set up model
    model = CGAN()
    model.to(device)
    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    # Set up diffusion
    diffusion = GaussianDiffusion(0.1, device)
    # Set up data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = datasets.ImageFolder(root='afhq/train', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    # Train model
    train(model, optimizer, dataloader, device, diffusion, 0.1)

if __name__ == '__main__':
    main()

## run the script in terminal
## python DCGAN.py