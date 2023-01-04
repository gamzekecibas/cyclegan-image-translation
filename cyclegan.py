### CYCLEGAN MODEL TRAINING WITH PYTORCH ###
### USING AFHQ DATASET BY DEFAULT ###
## TRAINING ON CPU BY DEFAULT ###
## implement wandb
## gpu available for macboook (M2)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import wandb
import matplotlib.pyplot as plt

device = torch.device("cuda")
print("device is working on: ", device)

import torch

if torch.cuda.is_available():
    print("GPU available")
else:
    print("GPU not available")


#wandb.init(project="CGAN", entity="comp511")

class CGAN(nn.Module):
    def __init__(self):
        super(CGAN, self).__init__()
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
        self.conv8 = nn.Conv2d(512, 512, 4, 2, 1, bias=False)
        self.deconv1 = nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=False)
        self.dbn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False)
        self.dbn2 = nn.BatchNorm2d(512)
        self.deconv3 = nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False)
        self.dbn3 = nn.BatchNorm2d(512)
        self.deconv4 = nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False)
        self.dbn4 = nn.BatchNorm2d(512)
        self.deconv5 = nn.ConvTranspose2d(1024, 256, 4, 2, 1, bias=False)
        self.dbn5 = nn.BatchNorm2d(256)
        self.deconv6 = nn.ConvTranspose2d(512, 128, 4, 2, 1, bias=False)
        self.dbn6 = nn.BatchNorm2d(128)
        self.deconv7 = nn.ConvTranspose2d(256, 64, 4, 2, 1, bias=False)
        self.dbn7 = nn.BatchNorm2d(64)
        self.deconv8 = nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = F.leaky_relu(self.conv1(x), 0.2)
        x2 = F.leaky_relu(self.bn2(self.conv2(x1)), 0.2)
        x3 = F.leaky_relu(self.bn3(self.conv3(x2)), 0.2)
        x4 = F.leaky_relu(self.bn4(self.conv4(x3)), 0.2)
        x5 = F.leaky_relu(self.bn5(self.conv5(x4)), 0.2)
        x6 = F.leaky_relu(self.bn6(self.conv6(x5)), 0.2)
        x7 = F.leaky_relu(self.bn7(self.conv7(x6)), 0.2)
        x8 = F.leaky_relu(self.conv8(x7), 0.2)
        x9 = F.dropout(F.relu(self.dbn1(self.deconv1(x8))), 0.5)
        x10 = torch.cat((x9, x7), 1)
        x11 = F.dropout(F.relu(self.dbn2(self.deconv2(x10))), 0.5)
        x12 = torch.cat((x11, x6), 1)
        x13 = F.dropout(F.relu(self.dbn3(self.deconv3(x12))), 0.5)
        x14 = torch.cat((x13, x5), 1)
        x15 = F.dropout(F.relu(self.dbn4(self.deconv4(x14))), 0.5)
        x16 = torch.cat((x15, x4), 1)
        x17 = F.relu(self.dbn5(self.deconv5(x16)))
        x18 = torch.cat((x17, x3), 1)
        x19 = F.relu(self.dbn6(self.deconv6(x18)))
        x20 = torch.cat((x19, x2), 1)
        x21 = F.relu(self.dbn7(self.deconv7(x20)))
        x22 = torch.cat((x21, x1), 1)
        x23 = self.tanh(self.deconv8(x22))
        return x23

def main():
    # Load dataset
    transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
    dataset = datasets.ImageFolder(root='afhq/train', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    # Initialize model
    model = CGAN()
    model = model.to(device)
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    ## WandB watch
    #wandb.watch(model, log="all")
    # Train model
    epochs = []
    losses = []
    for epoch in range(20):
        epoch_loss = 0
        epochs.append(epoch)
        for i, data in enumerate(dataloader, 0):
            # Get data
            real = data[0].to(device)
            # Forward pass
            fake = model(real)
            # Backward pass
            optimizer.zero_grad()
            loss = torch.mean(torch.abs(fake - real))
            epoch_loss += loss
            loss.backward()
            optimizer.step()
            # Print loss
            #wandb.log({"loss": loss, "epoch": epoch})
            
            #wandb.log({"real": wandb.Image(real)})
            #wandb.log({"fake": wandb.Image(fake)})
            print('[%d/%d][%d/%d] Loss: %.4f' % (epoch, 20, i, len(dataloader), loss.item()))
            # Save model
            if i % 100 == 0:
                torch.save(model.state_dict(), 'model.pth')
        losses.append(epoch_loss)

    plt.plot(epochs, epoch_loss)
    plt.show()
    

if __name__ == '__main__':
    main()

## run the script in terminal
## python cyclegan.py 