import torch
import  torch.nn as nn
import torch.functional as F
import os
import torchvision
from torchvision import datasets
from torchvision import transforms,utils
import numpy as np
from torchvision.utils import save_image,make_grid
import CGAN
import matplotlib.pyplot as plt
from train import train


batchsize = 4
z_dimension = 100
num = 25
epoch_num  = 5
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


    train_data = datasets.ImageFolder(r'./resized', transform=transforms.Compose([
        transforms.Resize((40, 30)),
        transforms.ToTensor(),
    ]))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batchsize, shuffle=True)

    #for i_batch, img in enumerate(train_loader):
     #   if i_batch == 0:
      #      print(img[1])  # 标签转化为编码
      #      fig = plt.figure()
      #      grid = utils.make_grid(img[0])
      #      plt.imshow(grid.numpy().transpose((1, 2, 0)))
      #      plt.show()
      #  break


    G = CGAN.generator(z_dimension,14400)
    D = CGAN.discriminator()

    g_optimizer = torch.optim.Adam(G.parameters(),lr = 1e-4)
    d_optimizer = torch.optim.Adam(D.parameters(),lr = 1e-5)

    train(d_optimizer=d_optimizer,g_optimizer = g_optimizer,
          dataloader = train_loader, epoch_num = epoch_num,
          G=G,D=D,criterion = criterion)

