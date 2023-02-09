import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from torchvision import models
from torchvision.models.inception import inception_v3

def cov(tensor, rowvar=True, bias=False):
    """Calculate a covariance matrix given data.
    
    Args:
        tensor (Tensor): A 1-D or 2-D tensor containing multiple variables and observations.
            Each row of `tensor` represents a variable, and each column a single observation of all
            those variables. Also see `rowvar` below.
        rowvar (bool, optional): If `True`, then each row represents a variable, with
            observations in the columns. Otherwise, the relationship is transposed: each column
            represents a variable, while the rows contain observations.
        bias (bool, optional): Default normalization (`False`) is by ``(N - 1)``, where ``N`` is the
            number of observations given (unbiased estimate). If `bias` is ``True``, then
            normalization is by ``N``. These values can be overridden by using the keyword
            arguments ``ddof=1`` and ``ddof=0`` respectively.

    Returns:
        Tensor: The covariance matrix of the variables in `tensor`.
    """
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * tensor @ tensor.transpose(-1, -2).conj()

def fid_metric(feature_model, real_samples, fake_samples):
    """
    Calculates the Fréchet Inception Distance (FID) to evalulate GANs
    Args:
        feature_model: Pretrained feature model that can extract features from images
        real_samples: Real samples from the dataset
        fake_samples: Fake (generated) samples

    Returns:
        The Frechet Inception Distance

    """
    real_activations = feature_model(real_samples)
    fake_activations = feature_model(fake_samples)

    # reshape the activations to 2D
    real_activations = real_activations.view(real_samples.size(0), -1)
    fake_activations = fake_activations.view(fake_samples.size(0), -1)
    # Step 3: Compute mean and covariance matrix
    real_mean = real_activations.mean(dim=0)
    # real_cov
    real_cov = cov(real_activations)


    fake_mean = fake_activations.mean(dim=0)
    fake_cov = cov(fake_activations)

    # Step 4: Calculate Fréchet distance
    trace = torch.trace(real_cov.contiguous().view(-1, real_cov.size(-1)) + fake_cov.contiguous().view(-1, fake_cov.size(-1)) - 2 * torch.mm(torch.sqrt(real_cov).contiguous().view(-1, real_cov.size(-1)), torch.sqrt(fake_cov).contiguous().view(-1, fake_cov.size(-1))))
    #print("trace:", trace.shape)
    fid = ((real_mean - fake_mean) ** 2).sum() + torch.trace(real_cov + fake_cov - 2 * torch.matmul(torch.sqrt(real_cov), torch.sqrt(fake_cov)))
    fid_l = (fid - 76) / 60
    return fid_l


inception_ = models.inception_v3(pretrained=True)


# Get the children of the model
children = list(inception_.children())

# Remove the output layer
model = torch.nn.Sequential(*children[:-1])

inception_.eval()

feature_model2= inception_.to('cuda:0')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((224,224)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

for real, fake in zip(real_samples, fake_samples):
  real_img = Image.open(real)
  fake_img = Image.open(fake)

  # Scale the pixel values
  real_img = (np.array(real_img) / 255) 
  fake_img = (np.array(fake_img) / 255)


  real_p = transform(real_img).to('cuda:0')
  fake_p = transform(fake_img).to('cuda:0')
  print(fid_metric(feature_model2, real_p.reshape(1,3,224,224).float(), fake_p.reshape(1,3,224,224).float() * 0.5 + 0.5))
  # break

  for real, fake in zip(real_samples, fake_samples):
  real_img = Image.open(real)
  fake_img = Image.open(fake)


  # Scale the pixel values
  real_img = (np.array(real_img) / 255) 
  fake_img = (np.array(fake_img) / 255)

  real_p = transform(real_img).to('cuda:0')
  fake_p = transform(fake_img).to('cuda:0')
  print(fid_metric(feature_model2, real_p.reshape(1,3,224,224).float(), fake_p.reshape(1,3,224,224).float() * 0.5 + 0.5))