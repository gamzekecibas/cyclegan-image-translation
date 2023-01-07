## STRUCTURE: DIFFUSION IMPLEMENTED CYCLEGAN##
### TASK: IMAGE TO IMAGE TRANSLATION (BY DEFAULT AFHQ CAT to AFHQ DOG) ###
### USING AFHQ DATASET BY DEFAULT ###
## TRAINING ON CPU BY DEFAULT ###
## TRAINING ON GPU IF AVAILABLE ###

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torchvision.utils import save_image

from tqdm import tqdm
import wandb

wandb.init(project="DCGAN", entity="gkecibas16")

class DCGAN(nn.Module):
    def __init__(self, img_channels, z_dim, feature_d, num_classes):
        super(DCGAN, self).__init__()
        self.gen = Generator(img_channels, z_dim, feature_d, num_classes)
        self.disc = Discriminator(img_channels, feature_d, num_classes)
        self.initialize_weights()

    def forward(self, x, y):
        return self.gen(x, y)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight.data, 1.0)
                nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self, img_channels, z_dim, feature_d, num_classes):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self.block(z_dim + num_classes, feature_d * 16, 4, 1, 0), # (N, feature_d*16, 4, 4)
            self.block(feature_d * 16, feature_d * 8, 4, 2, 1), # (N, feature_d*8, 8, 8)
            self.block(feature_d * 8, feature_d * 4, 4, 2, 1), # (N, feature_d*4, 16, 16)
            self.block(feature_d * 4, feature_d * 2, 4, 2, 1), # (N, feature_d*2, 32, 32)
            nn.ConvTranspose2d(feature_d * 2, img_channels, 4, 2, 1), # (N, img_channels, 64, 64)
            nn.Tanh(),
        )

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        return self.gen(x)

    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

class Discriminator(nn.Module):
    def __init__(self, img_channels, feature_d, num_classes):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.block(img_channels + num_classes, feature_d * 2, 4, 2, 1), # (N, feature_d*2, 32, 32)
            self.block(feature_d * 2, feature_d * 4, 4, 2, 1), # (N, feature_d*4, 16, 16)
            self.block(feature_d * 4, feature_d * 8, 4, 2, 1), # (N, feature_d*8, 8, 8)
            self.block(feature_d * 8, feature_d * 16, 4, 2, 1), # (N, feature_d*16, 4, 4)
            nn.Conv2d(feature_d * 16, 1, 4, 1, 0), # (N, 1, 1, 1)
        )

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        return self.disc(x)

    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight.data, 1.0)
            nn.init.constant_(m.bias.data, 0.0)

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

### GRADIENT CALCULATIONS MAY BE UPDATED !!
def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    mixed_images = real * epsilon + fake * (1 - epsilon)

    mixed_scores = critic(mixed_images)

    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def train_fn(loader, gen, disc, opt_gen, opt_disc, l1, mse, d_scaler, g_scaler, z_dim, device):
    loop = tqdm(loader)

    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(device)

        cur_batch_size = real.shape[0]
        ### TRAINING PART GRADIENT & LOSS CALCULATIONS MAY BE UPDATED !!
        # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))

        ### NOISE ADDITION TECHNIQUE SHOULD BE IMPROVED
        noise = torch.randn(cur_batch_size, z_dim, 1, 1).to(device)
        fake = gen(noise, real)

        disc_real = disc(real)
        disc_fake = disc(fake.detach())

        gp = gradient_penalty(disc, real, fake, device=device)
        loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake)) + gp * 10
        # loss_disc = -torch.mean(disc_real) + torch.mean(disc_fake)

        opt_disc.zero_grad()
        d_scaler.scale(loss_disc).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        output = disc(fake)
        loss_gen = -torch.mean(output)

        opt_gen.zero_grad()
        g_scaler.scale(loss_gen).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if batch_idx % 100 == 0:
            loop.set_postfix(
                d_loss=loss_disc.item(), g_loss=loss_gen.item()
            )

def main():
    # Hyperparameters etc.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 3e-4
    z_dim = 64
    image_channels = 3
    feature_d = 16
    num_classes = 10
    batch_size = 128
    num_epochs = 100
    cur_step = 0
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    cur_step = 0
    display_step = 500
    criterion = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()
    mse = nn.MSELoss()
    image_size = 64
    dataset = datasets.ImageFolder(
        root="afhq/train",
        transform=transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5 for _ in range(image_channels)], [0.5 for _ in range(image_channels)]
                ),
            ]
        ),
    )
    loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    gen = Generator(z_dim, image_channels, feature_d, num_classes).to(device)
    initialize_weights(gen)
    disc = Discriminator(image_channels, feature_d, num_classes).to(device)
    initialize_weights(disc)
    opt_gen = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_disc = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        train_fn(loader, gen, disc, opt_gen, opt_disc, l1, mse, d_scaler, g_scaler, z_dim, device)

    save_checkpoint(
        {
            "gen_state_dict": gen.state_dict(),
            "disc_state_dict": disc.state_dict(),
            "opt_gen_state_dict": opt_gen.state_dict(),
            "opt_disc_state_dict": opt_disc.state_dict(),
        },
        filename="my_checkpoint.pth.tar",
    )

if __name__ == "__main__":
    main()

## python DCGAN.py