import os
import numpy as np
import random
import math
import torch
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
manual_seed = 10
num_epochs = 5
random.seed(manual_seed)
torch.manual_seed(manual_seed)

#data paths
data_path_Train_A = os.path.dirname('afhq/train/cat')
data_path_Train_B = os.path.dirname('afhq/train/dog')
data_path_Test_A = os.path.dirname('afhq/val/cat')
data_path_Test_B = os.path.dirname('afhq/val/dog')

batch_size = 64  ## 128
#num_workers = 0 ## 2

transform = transforms.Compose([transforms.Resize((256,256)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,)),])
# cat train
load_Train_A = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(root=
               data_path_Train_A, transform=transform), batch_size=batch_size, 
               shuffle =True)
# dog train
load_Train_B = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(root=
               data_path_Train_B, transform=transform), batch_size=batch_size, 
               shuffle =True)
# cat test
load_Test_A = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(root=
              data_path_Test_A, transform=transform), batch_size=batch_size,
              shuffle = False)
# dog test
load_Test_B = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(root=
              data_path_Test_B, transform=transform), batch_size=batch_size,
              shuffle = False)
cat_train, _ = next(iter(load_Train_A))
dog_train, _ = next(iter(load_Train_B))
cat_test, _ = next(iter(load_Test_A))
dog_test, _ = next(iter(load_Test_B))


# define the generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, input):
        return self.main(input)

# define the discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input)

# initialize the generator and discriminator
netG = Generator().to(device)
netD = Discriminator().to(device)

# initialize the weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

netG.apply(weights_init)
netD.apply(weights_init)

# initialize the loss function
criterion = nn.MSELoss()

# initialize the optimizers
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

def train():
    # training loop
    for epoch in range(num_epochs):
        for i, (data_A, _) in enumerate(load_Train_A):
            # train the discriminator
            netD.zero_grad()
            real_A = data_A.to(device)
            # batch_size = real_A.size(0)
            label = torch.full((batch_size, 1, 13, 13), 1, device=device)
            output = netD(real_A)
            ## reshape label to have same shape with output
            #print('shape of output: ', output.shape)
            #print('shape of label: ', label.shape)
            ## return Long to float
            label = label.float()
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(batch_size, 100, 13, 13, device=device)
            fake_A = netG(noise)
            label.fill_(0)
            output = netD(fake_A.detach())
            label = label.float()
            ### reshape label to have same shape with output
            #print('shape of output: ', output.shape)
            #print('shape of label: ', label.shape)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # train the generator
            netG.zero_grad()
            label.fill_(1)
            output = netD(fake_A)
            label = label.float()
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if i % 10 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(load_Train_A),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # save the images
                vutils.save_image(real_A,
                        'real_samples.png',
                        normalize=True)
                fake = netG(noise)
                vutils.save_image(fake.detach(),
                        'fake_samples.png',
                        normalize=True)

                # log the images
                wandb.log({"real": [wandb.Image(real_A, caption="Real")],
                        "fake": [wandb.Image(fake.detach(), caption="Fake")]})

                # log the losses
                wandb.log({"errD": errD.item(), "errG": errG.item()})
                wandb.log({"D(x)": D_x, "D(G(z))": D_G_z1, "D(G(z))": D_G_z2})

# save the final model
torch.save(netG.state_dict(), 'netG_final.pth')
torch.save(netD.state_dict(), 'netD_final.pth')

if __name__ == '__main__':
    # initialize wandb
    wandb.init(project="CGAN_TRANSLATION", entity="gkecibas16")

    # train the model
    train()

    ## run the script in terminal
    ## python cgan_translation.py
    ## It is observable in wandb page