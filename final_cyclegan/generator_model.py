import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Residual block with instance normalization."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, img_channels, num_features=64, num_residuals=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect",
            ),
            nn.InstanceNorm2d(num_features),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.down_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        num_features, num_features * 2, kernel_size=3, stride=2, padding=1,
                    ),
                    nn.InstanceNorm2d(num_features * 2),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(
                        num_features * 2, num_features * 4, kernel_size=3, stride=2, padding=1,
                    ),
                    nn.InstanceNorm2d(num_features * 4),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
            ]
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )
        self.up_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(
                        num_features * 4, num_features * 2, kernel_size=3, stride=2, padding=1, output_padding=1,
                    ),
                    nn.InstanceNorm2d(num_features * 2),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(
                        num_features * 2, num_features * 1, kernel_size=3, stride=2, padding=1, output_padding=1,
                    ),
                    nn.InstanceNorm2d(num_features * 1),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
            ]
        )

        self.last = nn.Conv2d(
            num_features * 1, img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect",
        )

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))



def test():
    img_channels = 3
    img_size = 256
    x = torch.randn((2, img_channels, img_size, img_size))
    gen = Generator(img_channels, 9)
    print(gen(x).shape)


if __name__ == "__main__":
    test()
