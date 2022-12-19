import torch
from torch import nn, optim
import torchvision.datasets as dsets
from tqdm import tqdm
from model import VariationalAutoencoder
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

# CONFIG
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_DIM = 784
HIDDEN_DIM = 200
LATENT_DIM = 20
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 3e-4  # Karpathy's learning rate (constant)

# DATA
dataset = dsets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
model = VariationalAutoencoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_function = nn.BCELoss(reduction='sum')

# TRAIN
for epoch in range(NUM_EPOCHS):
    for batch_idx, (data, _) in enumerate(tqdm(train_loader)):
        data = data.to(DEVICE)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data.view(-1, INPUT_DIM)) + 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss.backward()
        train_loss = loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

model = model.to('cpu')
def inference(digit, num_examples = 1):
    images = []
    idx = 0
    for x,y in dataset:
        if y == idx:
            images.append(x)
            idx += 1
        if idx == 10:
            break
    
    encodings_digit = []
    for d in range(10):
        with torch.no_grad():
            