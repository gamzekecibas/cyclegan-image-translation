import argparse
import parser
from tkinter import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image

class Encoder(nn.Module):
    '''Encoder class for the VAE'''
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1)
        self.conv5 = nn.Conv2d(256, 512, 3, 1)
        self.conv6 = nn.Conv2d(512, 1024, 3, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(1024, 512)

    def forward(self, x):
        '''Forward pass of the encoder'''
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = x.view(-1, 1024)
        return self.fc1(x), self.fc2(x)

class Decoder(nn.Module):
    '''Decoder class for the VAE'''
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(512, 1024)
        self.conv1 = nn.ConvTranspose2d(1024, 512, 3, 1)
        self.conv2 = nn.ConvTranspose2d(512, 256, 3, 1)
        self.conv3 = nn.ConvTranspose2d(256, 128, 3, 1)
        self.conv4 = nn.ConvTranspose2d(128, 64, 3, 1)
        self.conv5 = nn.ConvTranspose2d(64, 32, 3, 1)
        self.conv6 = nn.ConvTranspose2d(32, 3, 3, 1)

    def forward(self, x):
        '''Forward pass of the decoder'''
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1024, 1, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return torch.sigmoid(self.conv6(x))

class Discriminator(nn.Module):
    '''Discriminator class for the VAE'''
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x):
        '''Forward pass of the discriminator'''
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(x))

class VAutoEncoder(nn.Module):
    '''Variational AutoEncoder class for image-to-image translation using Pytorch and Torchvision modules.
    Sample dataset: AFHQ dataset
    INPUTS: args - argparse object containing all the arguments for the model        
    '''
    def __init__(self, args):
        super(VAutoEncoder, self).__init__()
        self.args = args
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.discriminator = Discriminator(args)
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        self.loss = nn.BCELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        '''Forward pass of the model'''
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        '''Encode the input image'''
        return self.encoder(x)

    def decode(self, x):
        '''Decode the latent vector'''
        return self.decoder(x)

    def discriminate(self, x):
        '''Discriminate the image'''
        return self.discriminator(x)

    def train(self, train_loader):
        '''Train the model'''
        self.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch = self(data)
            loss = self.loss(recon_batch, data)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data)))
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

    def test(self, test_loader):
        '''Test the model'''
        self.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                data = data.to(self.device)
                recon_batch = self(data)
                test_loss += self.loss(recon_batch, data).item()
                if i == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n], recon_batch.view(self.args.batch_size, 3, 256, 256)[:n]])
                    save_image(comparison.cpu(), 'results/reconstruction_' + str(epoch) + '.png', nrow=n)
        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

def loss_function(recon_x, x, mu, logvar):
    '''Loss function for the VAE'''
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE CelebA Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed ( default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(datasets.CelebA('../data', download=True, transform=transforms.ToTensor()), batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(datasets.CelebA('../data', transform=transforms.ToTensor()), batch_size=args.test_batch_size, shuffle=True, **kwargs)
    if not args.no_cuda:
        encoder = Encoder(args).cuda()
        decoder = Decoder(args).cuda()
        discriminator = Discriminator(args).cuda()
    else:
        encoder = Encoder(args)
        decoder = Decoder(args)
        discriminator = Discriminator(args)
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=args.lr)
   
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        decoder.train()
        discriminator.train()
        train_loss = 0
       
       ## data should be tensor NO TUPLE
        for batch_idx, (data, _) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_discriminator.zero_grad()
            res_enc = encoder(data)
            recon_data = torch.tensor(res_enc[0], requires_grad=True)
            recon_batch = decoder(recon_data)
            loss = loss_function(recon_batch, data, encoder.mu, encoder.logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data)))
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
       
        encoder.eval()
        decoder.eval()
        discriminator.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                data = data.to(args.device)
                recon_batch = decoder(encoder(data))
                test_loss += loss_function(recon_batch, data, encoder.mu, encoder.logvar).item()
                if i == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n], recon_batch.view(args.batch_size, 3, 256, 256)[:n]])
                    save_image(comparison.cpu(), 'results/reconstruction_' + str(epoch) + '.png', nrow=n)
        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

        if args.save_model:
            torch.save(encoder.state_dict(), "encoder.pt")
            torch.save(decoder.state_dict(), "decoder.pt")
            torch.save(discriminator.state_dict(), "discriminator.pt")

## run vae_update.py script using appropriate arguments
## python vae_update.py --epochs 10 --batch-size 64 --lr 0.001 --log-interval 10 --no-cuda