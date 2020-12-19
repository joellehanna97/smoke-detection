import os
import pandas as pd
import rasterio as rio
import numpy as np
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


os.environ['CUDA_VISIBLE_DEVICES'] = '3'


class Net1(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net1, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x


class Net2(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net2, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden, 100)   # hidden layer
        self.hidden3 = torch.nn.Linear(100, 10)   # hidden layer
        self.predict = torch.nn.Linear(10, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden1(x))      # activation function for hidden layer
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
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


class TestFeaturesWeatherDataset(Dataset):
    def __init__(self, X=None):
        """
        Args:
            X (array)
        """
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Load data and get label
        X = self.X[idx]
        return X


def extract_images_feature(train_dl):
    all_outputs = []
    all_images = []
    # Create Model
    model = models.resnext50_32x4d(pretrained=True)

    model.conv1 = torch.nn.Conv2d(12, 64, kernel_size=(3, 3),
                                  stride=(2, 2), padding=(3, 3), bias=False)  # modify first layer

    new_classifier = nn.Sequential(*list(model.children())[:-1])  # remove last layer
    model.classifier = new_classifier

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(device)

    model.to(device)
    for epoch in range(5):  # loop over the dataset multiple times
        for i, data in enumerate(train_dl, 0):
            inputs = data['img'].float().to(device)

            # forward
            outputs = model(inputs)
            if epoch == 4:
                all_images.append(data['imgfile'])
                all_outputs.append(outputs.detach().numpy())
    print('Finished Training')

    PATH = '/netscratch/jhanna/smoke-detection/image_net.pth'
    torch.save(model.state_dict(), PATH)
    return all_outputs, all_images


def train_gen_output(train_r):
    model = Net2(n_feature=1004, n_hidden=200, n_output=1)     # define the network
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    losses = []
    model.to(device)
    # train the network
    all_outputs = []
    for epoch in range(70):
        for i, data in enumerate(train_r):
            batch_loss = []
            inputs, labels = data[0].float().to(device), data[1].float().to(device)
            # forward
            outputs = model(inputs)
            loss = loss_func(outputs, labels)     # must be (1. nn output, 2. target)
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
            batch_loss.append(loss.data)
            if epoch == 69:
                all_outputs.append(outputs.detach().numpy())
        losses.append(np.mean(batch_loss))
    print(losses)
    PATH = '/netscratch/jhanna/smoke-detection/gen_net.pth'
    torch.save(model.state_dict(), PATH)
    return all_outputs


def test_gen_output(test_r):
    model = Net2(n_feature=1004, n_hidden=200, n_output=1)     # define the network
    PATH = '/netscratch/jhanna/smoke-detection/gen_net.pth'
    model.load_state_dict(torch.load(PATH))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i, data in enumerate(test_r):
        inputs = data.float().to(device)
        outputs = model(inputs)
    np.save('predictions', outputs)


def main():
    # Extract Image Features
    image_data = SmokePlumesSubsetDataset(datadir='/netscratch/jhanna/images_subset/training/')
    train_fe = torch.utils.data.DataLoader(image_data, batch_size=64)  # data loader
    output, images = extract_images_feature(train_fe)

    # Add Weather Data
    df = pd.read_csv('/netscratch/jhanna/labels.csv')
    df.filename = df.filename.str.replace(':', '-')
    inputs = []
    labels = []
    for i, batch in enumerate(output):
        for j in range(0, len(batch)):
            current_img = images[i][j].split('/')[6]
            weather = df[df['filename'] == current_img][['temp', 'humidity', 'wind-u', 'wind-v']].to_numpy()
            feats = output[i][j]
            inputs.append(np.append(weather, feats))
            labels.append(df[df['filename'] == current_img]['gen_output'].values)
    labels = np.array(labels)

    # Predict Generation Output
    # inputs = (inputs - np.mean(inputs)) / (np.std(inputs))
    inputs = inputs / np.max(inputs)
    gen_data = FeaturesWeatherDataset(X=inputs, y=labels)
    train_r = torch.utils.data.DataLoader(gen_data, batch_size=64)
    train_gen_output(train_r)

    # TEST

    # Extract Image Features
    image_data = SmokePlumesSubsetDataset(datadir='/netscratch/jhanna/images_subset/testing/')
    test_fe = torch.utils.data.DataLoader(image_data, batch_size=64)  # data loader
    output, images = extract_images_feature(test_fe)

    # Add Weather Data
    df = pd.read_csv('/netscratch/jhanna/test_labels.csv')
    df.filename = df.filename.str.replace(':', '-')
    inputs = []
    labels = []
    for i, batch in enumerate(output):
        for j in range(0, len(batch)):
            current_img = images[i][j].split('/')[6]
            weather = df[df['filename'] == current_img][['temp', 'humidity', 'wind-u', 'wind-v']].to_numpy()
            feats = output[i][j]
            inputs.append(np.append(weather, feats))
            labels.append(df[df['filename'] == current_img]['gen_output'].values)
    labels = np.array(labels)

    inputs = inputs / np.max(inputs)
    gen_data = TestFeaturesWeatherDataset(X=inputs)
    np.save('/netscratch/jhanna/smoke-detection/labels', labels)
    test_r = torch.utils.data.DataLoader(gen_data, batch_size=64)
    test_gen_output(test_r)


if __name__ == "__main__":
    main()
