## referance links
## referance links
import os
import random
import torch
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import wandb
import fid_metric as fid

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
manual_seed = 10
num_epochs = 100
decay_start_epoch = 5
random.seed(manual_seed)
torch.manual_seed(manual_seed)

#data paths
## instead of using os.path.dirname, use os.path.join
main_path = os.path.dirname('afhq')
data_path_Train_A = torch.ImageFolder(os.path.join(main_path, 'train/cat'))
data_path_Train_B = os.path.join(main_path, 'train/dog')
data_path_Test_A = os.path.join(main_path, 'val/cat')
data_path_Test_B = os.path.join(main_path, 'val/dog')

batch_size = 1  ## 128
num_workers = 2

transform = transforms.Compose([transforms.Resize((64,64)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,))])
# cat train
load_Train_A = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(root=
               data_path_Train_A, transform=transform), batch_size=batch_size, 
               shuffle =True, num_workers=num_workers)
# dog train
load_Train_B = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(root=
               data_path_Train_B, transform=transform), batch_size=batch_size, 
               shuffle =True, num_workers=num_workers)
# cat test
load_Test_A = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(root=
              data_path_Test_A, transform=transform), batch_size=batch_size,
              shuffle = False, num_workers=num_workers)
# dog test
load_Test_B = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(root=
              data_path_Test_B, transform=transform), batch_size=batch_size,
              shuffle = False, num_workers=num_workers)

cat_train, _ = next(iter(load_Train_A))
dog_train, _ = next(iter(load_Train_B))
cat_test, _ = next(iter(load_Test_A))
dog_test, _ = next(iter(load_Test_B))

# define the generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(3, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, input):
        return self.main(input)


# gradient penalty to stabilize the training considering dimension of the image
## not use for now
def gradient_penalty(critic, real, fake, device=device):
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * epsilon + fake * (1 - epsilon)
    mixed_scores = critic(interpolated_images)
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

# # define the discriminator
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
        nn.init.normal_(m.weight.data, 0.001, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

netG.apply(weights_init)
netD.apply(weights_init)

# initialize the loss function for discriminator and generator
criterion_gan = nn.MSELoss()
criterion_l1 = torch.nn.L1Loss()

# initialize the optimizers
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.8, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.8, 0.999))

def loss_D(real, fake):
    loss_real = criterion_gan(real, torch.ones_like(real))
    loss_fake = criterion_gan(fake, torch.zeros_like(fake))
    return (loss_real + loss_fake) / 2

def loss_G(fake):
    return criterion_gan(fake, torch.ones_like(fake))

## calculate FID score for evaluation
def calculate_fid(real, fake):
    ## input real_images and fake_images (they are torch tensors in cuda)
    ## output FID score
    
    ## calculate the mean and covariance matrix of real images
    mu_real = torch.mean(real, dim=0)
    cov_real = torch.zeros(3, 3).to(device)
    for i in range(real.shape[0]):
        cov_real += torch.mm((real[i] - mu_real).unsqueeze(0).t(), (real[i] - mu_real).unsqueeze(0))
    cov_real /= real.shape[0]

    ## calculate the mean and covariance matrix of fake images
    mu_fake = torch.mean(fake, dim=0)
    cov_fake = torch.zeros(3, 3).to(device)
    for i in range(fake.shape[0]):
        cov_fake += torch.mm((fake[i] - mu_fake).unsqueeze(0).t(), (fake[i] - mu_fake).unsqueeze(0))
    cov_fake /= fake.shape[0]

    ## calculate the FID score
    fid = torch.trace(cov_real + cov_fake - 2 * torch.sqrt(torch.mm(cov_real, cov_fake)))
    return fid

## create a train function to translate train_A to train_B
def train():
    for epoch in range(num_epochs):
        iteration = 0
        for i, data in enumerate(zip(load_Train_A, load_Train_B), 0):
            # get the inputs
            ## real_A & real_B should be required grad = True
            real_A = data[0][0].to(device)
            real_B = data[1][0].to(device)
            real_A.requires_grad = True
            real_B.requires_grad = True

            # train the discriminator
            netD.zero_grad()
            # train with real images
            output = netD(real_B)
            #errD_real = criterion_gan(output, torch.ones_like(output))
            D_x = output.mean().item()
            # train with fake images
            #grad_penalty = 1e-4 * gradient_penalty(netD, real_B, real_A, device)
            fake_B = netG(real_A)
            ### transform fake_B 1072x1072 to 64x64
            fake_B = fake_B[:, :, 0:64, 0:64]
            output = netD(fake_B.detach())
            #errD_fake = criterion_gan(output, torch.zeros_like(output))
            D_G_z1 = output.mean().item()
            # total loss
            errD = loss_D(output, fake_B.detach())
            errD.backward()
            optimizerD.step()

            # train the generator
            netG.zero_grad()
            output = netD(fake_B)
            errG = loss_G(output)
            D_G_z2 = output.mean().item()
            errG.backward()
            optimizerG.step()

            iteration += 1
            if (iteration % 1000 == 0):
                
                # save the images
                vutils.save_image(real_A,
                        'real_samples.png',
                        normalize=True)
                fake = netG(real_A)
                vutils.save_image(fake.detach(),
                        'fake_samples.png',
                        normalize=True)

                # log the images
                wandb.log({"real_samples": [wandb.Image('real_samples.png', caption="Real Samples")],
                            "fake_samples": [wandb.Image('fake_samples.png', caption="Fake Samples")]})
                ## print path of real_B
                print('real_B path: ', data_path_Train_B[i])
                # log the losses
                wandb.log({"Loss_D": errD.item(),
                            "Loss_G": errG.item()})
                wandb.log({"D(x)": D_x,
                            "D(G(z1))": D_G_z1,
                            "D(G(z2))": D_G_z2})
                ## save fid & kid scores
                fid_score = fid(real_B, fake_B)
                #kid = calculate_kid(real_B, fake_B)
                wandb.log({"FID": fid_score})

                #wandb.log({'grad_penalty': grad_penalty})
                wandb.log({"epoch": epoch})

                # save the final model
                torch.save(netG.state_dict(), 'netG_final.pth')
                torch.save(netD.state_dict(), 'netD_final.pth')


if __name__ == '__main__':
    torch.cuda.empty_cache()
    # initialize wandb
    wandb.init(project="CGAN_TRANSLATION", entity="comp511")

    # train the model
    train()

    ## run the script in terminal
    ## python cgan_translation.py
    ## It is observable in wandb page