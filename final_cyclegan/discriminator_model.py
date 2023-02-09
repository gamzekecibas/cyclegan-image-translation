import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                nn.Sequential( 
                    nn.Conv2d(
                        in_channels, feature, 4,
                        stride=1 if feature == features[-1] else 2, 
                        padding = 1, bias=True, padding_mode="reflect",
                    ),
                    nn.InstanceNorm2d(feature),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            in_channels = feature
        layers.append(
            nn.Conv2d(
                in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect",
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        return torch.sigmoid(self.model(x))


def test():
    x = torch.randn((5, 3, 256, 256))
    model = Discriminator(in_channels=3)
    preds = model(x)
    print(preds.shape)


if __name__ == "__main__":
    test()
