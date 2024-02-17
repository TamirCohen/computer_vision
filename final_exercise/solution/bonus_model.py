"""Define your architecture here."""
import torch
from models import SimpleNet
from torch import nn
import torch.nn.functional as F
from xcpetion import Block, SeparableConv2d
import math

class LilXception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=2):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(LilXception, self).__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False,
                            grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True,
                            grow_first=True)
        self.block3 = Block(256, 400, 2, 2, start_with_relu=True,
                            grow_first=True)

        self.block4 = Block(400, 400, 3, 1, start_with_relu=True,
                            grow_first=True)
        self.block5 = Block(400, 400, 3, 1, start_with_relu=True,
                            grow_first=True)
        self.block6 = Block(400, 400, 3, 1, start_with_relu=True,
                            grow_first=True)
        self.block7 = Block(400, 400, 3, 1, start_with_relu=True,
                            grow_first=True)

        self.block8 = Block(400, 400, 3, 1, start_with_relu=True,
                            grow_first=True)
        self.block9 = Block(400, 400, 3, 1, start_with_relu=True,
                            grow_first=True)
        self.block10 = Block(400, 400, 3, 1, start_with_relu=True,
                             grow_first=True)
        self.block11 = Block(400, 400, 3, 1, start_with_relu=True,
                             grow_first=True)

        self.block12 = Block(400, 800, 2, 2, start_with_relu=True,
                             grow_first=False)

        self.conv3 = SeparableConv2d(800, 1000, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1000)

        # do relu here
        self.conv4 = SeparableConv2d(1000, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc1 = nn.Linear(2048, 1000)
        self.fc2 = nn.Linear(1000, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, num_classes)

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # -----------------------------

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)

        return x


class DeepSimpleNet(nn.Module):
    """Simple Convolutional and Fully Connect network."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=(7, 7))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(7, 7))
        self.conv3 = nn.Conv2d(16, 24, kernel_size=(7, 7))
        self.fc1 = nn.Linear(24 * 26 * 26, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, image):
        """Compute a forward pass."""
        first_conv_features = self.pool(F.relu(self.conv1(image)))
        second_conv_features = self.pool(F.relu(self.conv2(
            first_conv_features)))
        third_conv_features = self.pool(F.relu(self.conv3(
            second_conv_features)))
        # flatten all dimensions except batch
        flattened_features = torch.flatten(third_conv_features, 1)
        fully_connected_first_out = F.relu(self.fc1(flattened_features))
        fully_connected_second_out = F.relu(self.fc2(fully_connected_first_out))
        two_way_output = self.fc3(fully_connected_second_out)
        return two_way_output

def my_bonus_model():
    """Override the model initialization here.

    Do not change the model load line.
    """
    # initialize your model:
    model = LilXception()
    # load your model using exactly this line (don't change it):
    model.load_state_dict(torch.load('checkpoints/bonus_model.pt')['model'])
    return model
