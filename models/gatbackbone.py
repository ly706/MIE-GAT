import os
import sys
import torch
import torch.nn as nn
import torchvision

current_path = os.getcwd()
sys.path.append(current_path)
sys.path.append(os.path.join(sys.path[0], ".."))
from config.data_config import config
cf = config()

# encoder for slice image
class SliceEmbeddingImagenet(nn.Module):
    def __init__(self, emb_size, depth=2, heads=2, pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super(SliceEmbeddingImagenet, self).__init__()
        # set size
        self.hidden = 32
        self.last_hidden = self.hidden * 4
        self.emb_size = emb_size

        # set layers
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                              out_channels=self.hidden,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=self.hidden,
                                              out_channels=int(self.hidden * 1.5),
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=int(self.hidden * 1.5)),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=int(self.hidden * 1.5),
                                              out_channels=self.hidden * 2,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 2),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.4))
        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels=self.hidden * 2,
                                              out_channels=self.hidden * 4,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 4),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.5))

        self.layer_last = nn.Sequential(
            nn.Linear(in_features=int(self.hidden * 4 * (cf.crop_size / 16) * (cf.crop_size / 16)),
                      out_features=self.emb_size, bias=True),
            nn.BatchNorm1d(self.emb_size))

        self.fc = nn.Linear(in_features=self.emb_size, out_features=2)

    def forward(self, input_img_data, input_struct=None):
        output_data = self.conv_1(input_img_data)
        output_data = self.conv_2(output_data)
        output_data = self.conv_3(output_data)
        output_data = self.conv_4(output_data)

        output_data = self.layer_last(output_data.view(output_data.size(0), -1))
        return output_data, self.fc(output_data)