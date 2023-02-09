import torch
from dataset import CatDogDataset
import sys
from torch._C import _dispatch_tls_is_dispatch_key_excluded
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator
import wandb 
from fid_metric import fid_metric as fid

def train_fn(
    disc_C, disc_D, gen_D, gen_C, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
):
    C_reals = 0
    C_fakes = 0
    loop = tqdm(loader, leave=True)

    vgg16 = models.vgg16(pretrained=True)

    # remove the last layers (classifier)
    vgg16 = torch.nn.Sequential(*list(vgg16.features)[:-1], vgg16.avgpool)
    vgg16.eval()
    feature_model = vgg16.to(config.DEVICE)

    for idx, (dog, cat) in enumerate(loop):
        dog = dog.to(config.DEVICE)
        cat = cat.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_cat = gen_C(dog)
            D_C_real = disc_C(cat)
            D_C_fake = disc_C(fake_cat.detach())
            C_reals += D_C_real.mean().item()
            C_fakes += D_C_fake.mean().item()
            D_C_real_loss = mse(D_C_real, torch.ones_like(D_C_real))
            D_C_fake_loss = mse(D_C_fake, torch.zeros_like(D_C_fake))
            D_C_loss = D_C_real_loss + D_C_fake_loss

            fake_dog = gen_D(cat)
            D_D_real = disc_D(dog) 
            D_D_fake = disc_D(fake_dog.detach())
            D_D_real_loss = mse(D_D_real, torch.ones_like(D_D_real))
            D_D_fake_loss = mse(D_D_fake, torch.zeros_like(D_D_fake))
            D_D_loss = D_D_real_loss + D_D_fake_loss

            # put it togethor
            D_loss = (D_C_loss + D_D_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_C_fake = disc_C(fake_cat)
            D_D_fake = disc_D(fake_dog)
            loss_G_C = mse(D_C_fake, torch.ones_like(D_C_fake))
            loss_G_D = mse(D_D_fake, torch.ones_like(D_D_fake))

            # cycle loss
            cycle_dog = gen_D(fake_cat)
            cycle_cat = gen_C(fake_dog)
            cycle_dog_loss = l1(dog, cycle_dog)
            cycle_cat_loss = l1(cat, cycle_cat)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_dog = gen_D(dog)
            identity_cat = gen_C(cat)
            identity_dog_loss = l1(dog, identity_dog)
            identity_cat_loss = l1(cat, identity_cat)

            # add all togethor
            G_loss = (
                loss_G_D
                + loss_G_C
                + cycle_dog_loss * config.LAMBDA_CYCLE
                + cycle_cat_loss * config.LAMBDA_CYCLE
                + identity_cat_loss * config.LAMBDA_IDENTITY
                + identity_dog_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_cat * 0.5 + 0.5, f"saved_images/cat_{idx}.png")
            save_image(fake_dog * 0.5 + 0.5, f"saved_images/dog_{idx}.png")

            #fid_score_dog = fid(feature_model, dog.float(), fake_dog * 0.5 + 0.5)
            #fid_score_cat = fid(feature_model, cat.float(), fake_cat * 0.5 + 0.5)
            #fid_score = (fid_score_dog + fid_score_cat) / 2

            wandb.log({"D_loss": D_loss})
            wandb.log({"G_loss": G_loss})
            wandb.log({"cycle_dog_loss": cycle_dog_loss})
            wandb.log({"cycle_cat_loss": cycle_cat_loss})
            wandb.log({"identity_cat_loss": identity_cat_loss})
            wandb.log({"identity_dog_loss": identity_dog_loss})
            wandb.log({"fake_cat": [wandb.Image(fake_cat)]})
            wandb.log({"fake_dog": [wandb.Image(fake_dog)]})
            wandb.log({"real_cat": [wandb.Image(cat)]})
            wandb.log({"real_dog": [wandb.Image(dog)]})
        

        loop.set_postfix(C_real=C_reals / (idx + 1), C_fake=C_fakes / (idx + 1))


def main():

    disc_C = Discriminator(in_channels=3).to(config.DEVICE)
    disc_D = Discriminator(in_channels=3).to(config.DEVICE)
    gen_D = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_C = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_C.parameters()) + list(disc_D.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_D.parameters()) + list(gen_C.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_H,
            gen_C,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_Z,
            gen_D,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_H,
            disc_C,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_Z,
            disc_D,
            opt_disc,
            config.LEARNING_RATE,
        )

    dataset = CatDogDataset(
        root_cat=config.TRAIN_DIR + "/cat",
        root_dog=config.TRAIN_DIR + "/dog",
        transform=config.transforms,
    )
    val_dataset = CatDogDataset(
        root_cat="cyclegan_test/cat1",
        root_dog="cyclegan_test/dog1",
        transform=config.transforms,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc_C,
            disc_D,
            gen_D,
            gen_C,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
        )

        if config.SAVE_MODEL:
            save_checkpoint(gen_C, opt_gen, filename=config.CHECKPOINT_GEN_C)
            save_checkpoint(gen_D, opt_gen, filename=config.CHECKPOINT_GEN_D)
            save_checkpoint(disc_C, opt_disc, filename=config.CHECKPOINT_CRITIC_C)
            save_checkpoint(_dispatch_tls_is_dispatch_key_excluded, opt_disc, filename=config.CHECKPOINT_CRITIC_D)

    
if __name__ == "__main__":
    wandb.init(project="CGAN_TRANSLATION", entity="comp511")
    main()