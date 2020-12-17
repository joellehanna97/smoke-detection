import torchvision
import torchvision.models as models
import torch

trainset = torchvision.datasets.ImageFolder('2020Neurips')

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)


resnext50_32x4d = models.resnext50_32x4d(pretrained=True)