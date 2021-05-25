import torch
import  torch.nn as nn
import torch.functional as F
import os
import torchvision
from torchvision import datasets

from torchvision import transforms
import numpy as np
from torchvision.utils import save_image,make_grid
import CGAN
from train import Generate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#超参数
batchsize = 128
z_dimension = 100
num = 25
epoch_num  = 100
scale = 1
criterion = nn.BCELoss()

if __name__ =="__main__":
    if not os.path.exists('./img'):
        os.mkdir('./img')
    if not os.path.exists('./fake'):
        os.mkdir('./fake')
    for i in range(0,22):
        if not os.path.exists('./fake/{}'.format(i)):
            os.mkdir('./fake/{}'.format(i))
    G = CGAN.generator(z_dimension,3136)
    G.load_state_dict(torch.load("generator.pth"))
    Generate(epoch_num, G)

