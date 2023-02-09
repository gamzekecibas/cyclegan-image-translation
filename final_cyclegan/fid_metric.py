import torch

def cov(tensor, rowvar=True, bias=False):
    """Estimate a covariance matrix (np.cov)"""
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * tensor @ tensor.transpose(-1, -2).conj()

def fid_metric(feature_model, real_samples, fake_samples):
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
    #trace = torch.trace(real_cov.contiguous().view(-1, real_cov.size(-1)) + fake_cov.contiguous().view(-1, fake_cov.size(-1)) - 2 * torch.mm(torch.sqrt(real_cov).contiguous().view(-1, real_cov.size(-1)), torch.sqrt(fake_cov).contiguous().view(-1, fake_cov.size(-1))))
    #print("trace:", trace.shape)
    fid = ((real_mean - fake_mean) ** 2).sum() + torch.trace(real_cov + fake_cov - 2 * torch.mm(torch.sqrt(real_cov), torch.sqrt(fake_cov)))
    return fid

    