import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2 

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


transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToTensor()
])


if __name__ == '__main__':

    model = CGAN()
    print(model)

    PATH = 'comp511_weights_20_epoch.pth'

    model.load_state_dict(torch.load(PATH))
    model.eval()

    test_path = "D:/masa üstü/Hamza Proje Dosyalar/COMP511_project/comp511-project-develop_vae/comp511-project-develop_vae/cyclegan-tutorial/afhq/val"
    test_set = torchvision.datasets.ImageFolder(root= test_path, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)

    for i, data in enumerate(test_loader):
        image, label = data
        print(image.shape)
        print(label)
        output = model(image)
        print(output.shape)
        """
        # save output image
        output = output.detach().numpy()
        output = np.transpose(output, (0, 2, 3, 1))
        output = (output + 1) / 2.0 * 255.0
        output = output.astype("uint8")

        image = image.detach().numpy()
        image = np.transpose(image, (0, 2, 3, 1))
        image = (image + 1) / 2.0 * 255.0
        image = image.astype("uint8")
        """
        filename = "output_image_{}.png".format(i)
        #cv2.imwrite(filename, output[0])
        save_image(output[0], filename)

        filename_input = "input_image_{}.png".format(i)
        #cv2.imwrite(filename_input, image[0])
        save_image(image[0], filename_input)

        if i==30:
            break
    
    loss = [0.68, 0.02]
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(loss)
    plt.show()