import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import pandas as pd
import os
import rasterio as rio
import numpy as np


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x


class SmokePlumesSubsetDataset(Dataset):
    """Smoke plumes subset dataset."""

    def __init__(self, datadir=None):
        """
        Args:
            datadir (string): Path to the folder of the images.
        """
        self.datadir = datadir

        self.imgfiles = []  # list of image files

        # read in image file names
        for root, dirs, files in os.walk(datadir):
            for filename in files:
                if not filename.endswith('.tif'):
                    # ignore files that or not GeoTIFFs
                    continue
                self.imgfiles.append(os.path.join(root, filename))

        # turn lists into arrays
        self.imgfiles = np.array(self.imgfiles)

    def __len__(self):
        return len(self.imgfiles)

    def __getitem__(self, idx):
        # read in data file
        imgfile = rio.open(self.imgfiles[idx])
        imgdata = np.array([imgfile.read(i) for i in
                            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]], dtype=np.float16)
        # skip band 11 (Sentinel-2 Band 10, Cirrus) as it does not contain
        # useful information in the case of Level-2A data products

        sample = {'idx': idx,
                  'img': imgdata,
                  'imgfile': self.imgfiles[idx]}

        return sample


class FeaturesWeatherDataset(Dataset):
    def __init__(self, X=None, y=None):
        """
        Args:
            X (array)
            y (array)
        """
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Load data and get label
        X = self.X[idx]
        y = self.y[idx]
        return X, y


def extract_images_feature(train_dl):
    # Create Model
    model = models.resnext50_32x4d(pretrained=True)

    model.conv1 = torch.nn.Conv2d(12, 64, kernel_size=(3, 3),
                                  stride=(2, 2), padding=(3, 3), bias=False)  # modify first layer

    new_classifier = nn.Sequential(*list(model.children())[:-1])  # remove last layer
    model.classifier = new_classifier
    # model.to(device)

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    for epoch in range(2):  # loop over the dataset multiple times
        for i, data in enumerate(train_dl, 0):
            # inputs, labels = data[0].to(device), data[1].to(device)
            inputs = data['img']
            # forward
            outputs = model(inputs)
            print(outputs)

    print('Finished Training')

    PATH = './cifar_net.pth'
    torch.save(model.state_dict(), PATH)
    return outputs


def train_gen_output(train_r):
    model = Net(n_feature=1, n_hidden=10, n_output=1)     # define the network
    # print(net)  # net architecture
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

    # train the network
    for epoch in range(2):
        for i, data in enumerate(train_r):
            inputs = data['X']
            # forward
            outputs = model(inputs)
            loss = loss_func(outputs, data['y'])     # must be (1. nn output, 2. target)

            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients


if __name__ == 'main':
    # Extract Image Features
    image_data = SmokePlumesSubsetDataset(datadir='/netscratch/jhanna/images_subset/training/')
    train_fe = torch.utils.data.DataLoader(image_data, batch_size=64)  # data loader
    output = extract_images_feature(train_fe)

    # Add Weather Data
    df = pd.read_csv('labels.csv')
    weather_data = df[['temp', 'humidity', 'wind-u', 'wind-v']].to_numpy()
    X = np.append(output, weather_data)
    y = df['gen_output'].to_numpy()

    # Predict Generation Output
    gen_data = FeaturesWeatherDataset(X=X, y=y)
    train_r = torch.utils.data.DataLoader(gen_data, batch_size=64)  # data loader
    train_gen_output(train_r)
