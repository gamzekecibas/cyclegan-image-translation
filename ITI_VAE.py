import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(512, self.args.z_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(-1, 512)
        x = self.fc1(x)
        return x

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(self.args.z_dim, 512)
        self.conv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 512, 1, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = torch.sigmoid(self.conv5(x))
        return x

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(-1, 512)
        x = torch.sigmoid(self.fc1(x))
        return x

class ITI_VAE(nn.Module):
    'Variational Auto Encoder class for image-to-image translation using Pytorch'

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.build_model()

    def build_model(self):
        super(ITI_VAE, self).__init__()
        self.encoder = Encoder(self.args).to(self.device)
        self.decoder = Decoder(self.args).to(self.device)
        self.discriminator = Discriminator(self.args).to(self.device)
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.args.lr)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.args.lr)
        self.criterion = nn.MSELoss()

    def train(self, train_loader):
        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            self.optimizer_D.zero_grad()
            z = self.encoder(data)
            x_hat = self.decoder(z)
            D_real = self.discriminator(data)
            D_fake = self.discriminator(x_hat)
            D_loss = self.criterion(D_real, torch.ones_like(D_real)) + self.criterion(D_fake, torch.zeros_like(D_fake))
            D_loss.backward()
            self.optimizer_D.step()
            G_loss = self.criterion(D_fake, torch.ones_like(D_fake)) + self.criterion(x_hat, data)
            G_loss.backward()
            self.optimizer.step()
            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), G_loss.item()))

    def test(self, test_loader):
        self.encoder.eval()
        self.decoder.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(self.device)
                z = self.encoder(data)
                x_hat = self.decoder(z)
                test_loss += self.criterion(x_hat, data).item()

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

    def save_model(self, epoch):
        torch.save(self.encoder.state_dict(), 'encoder_{}.pth'.format(epoch))
        torch.save(self.decoder.state_dict(), 'decoder_{}.pth'.format(epoch))
        torch.save(self.discriminator.state_dict(), 'discriminator_{}.pth'.format(epoch))

    def load_model(self, epoch):
        self.encoder.load_state_dict(torch.load('encoder_{}.pth'.format(epoch)))
        self.decoder.load_state_dict(torch.load('decoder_{}.pth'.format(epoch)))
        self.discriminator.load_state_dict(torch.load('discriminator_{}.pth'.format(epoch)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ITI_VAE MNIST Example')
    parser.add_argument('--dataset', type=str, default='mnist', metavar='N', help='dataset to use')
    parser.add_argument('--z_dim', type=int, default=20, metavar='N', help='dimension of latent variable')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_step', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    ## run the script with --no-cuda to disable GPU
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader( datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor()), batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader( datasets.MNIST('../data', train=False, transform=transforms.ToTensor()), batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = ITI_VAE(args)
    for epoch in range(1, args.epochs + 1):
        model.train(train_loader)
        model.test(test_loader)
        model.save_model(epoch)

    model.load_model(args.epochs)
    model.test(test_loader)

    with torch.no_grad():
        sample = torch.randn(64, args.z_dim).to(model.device)
        sample = model.decoder(sample).cpu()
        save_image(sample.view(64, 1, 28, 28), 'sample_' + str(args.epochs) + '.png')

        ## Run the script in terminal
        ## python ITI_VAE.py --dataset mnist --z_dim 20 --batch-size 128 --test-batch-size 1000 --epochs 10 --lr 0.001 --no-cuda