import torch
import  torch.nn as nn
import torch.functional as F
import os
import torchvision
from torchvision import datasets
import matplotlib.pyplot as plt

from torchvision import transforms,utils
import numpy as np
from torchvision.utils import save_image,make_grid
import CGAN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#超参数
batchsize = 4
z_dimension = 100
num = 25
epoch_num  = 5
scale = 1
criterion = nn.BCELoss()

def Generate(num, G):
    G.to(device)
    step = 0
    for i in range(0,22):
        print(i)
        for epoch in range(num):
            z = torch.Tensor(torch.randn(1, z_dimension)).to(device)
            fake_labels = generatelabels(batchsize)
            fake_img = G(z, fake_labels)
            grid = utils.make_grid(fake_img.cpu().data)
            plt.axis('off')
            plt.set_size_inches(40/100.0, 30/100.0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
            plt.margins(0,0)
            plt.imshow(grid.numpy().transpose((1, 2, 0)))
            plt.show()
            plt.savefig('./fake/{}/fake_images-{}.png'.format(i,num))
            plt.close


def train(d_optimizer, g_optimizer, dataloader, epoch_num, G, D, criterion):
    '''

    :param d_optimizer:
    :param g_optimizer:
    :param dataloader:
    :param epoch_num:
    :param G:
    :param D:
    :param criterion:
    :return:
    '''
    G.to(device)
    D.to(device)
    # g_optimizer.to(device)
    # d_optimizer.to(device)
    # criterion.to(device)
    step = 0

    for epoch in range(epoch_num):
        for i, (imgs, real_labels) in enumerate(dataloader):
            num_img = imgs.size(0)
            real_label = torch.Tensor(torch.ones(num_img)).to(device)
            fake_label = torch.Tensor(torch.zeros(num_img)).to(device)

            real_labels = generatelabels(batchsize, real_labels)  # 产生对应的one-hot编码标签
            real_labels.requires_grad = True

            imgs = imgs.to(device)
            num_img = imgs.size(0)
            real_out = D(imgs, real_labels)  # 输入真实图片得到结果
            real_scores = real_out
            d_loss_real = criterion(real_out, real_label)

            z = torch.Tensor(torch.randn(num_img, z_dimension)).to(device)

            fake_labels = generatelabels(batchsize)  # 生成编码标签
            fake_labels.requires_grad = True

            z.requires_grad = True
            fake_img = G(z, fake_labels)
            fake_out = D(fake_img, fake_labels)

            d_loss_fake = criterion(fake_out, fake_label)

            fake_scores = fake_out  #

            # 先更新判别器参数 然后再更新生成器参数
            d_loss = d_loss_real + d_loss_fake

            # 第一个epoch先充分训练判别器 所以每十次迭代才更新一次生成器
            # if epoch == 0:
            d_optimizer.zero_grad()  # 梯度清零
            d_loss.backward()  # 计算梯度
            d_optimizer.step()  # 更新参数

                # # 更新生成器
                # if i % 10 == 0:
            z = torch.Tensor(torch.randn(num_img, z_dimension)).to(device)
            z.requires_grad = True
            fake_img = G(z, fake_labels)
            fake_out = D(fake_img, fake_labels)
            g_loss = criterion(fake_out, real_label)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            fake_images = to_img(fake_img.cpu().data)

            grid = make_grid(fake_images, nrow=4, padding=0)


            # else:  # 后面的迭代每隔25次迭代才更新一次判别器
            #     if i % num == 0:
            #         d_optimizer.zero_grad()  # 梯度清零
            #         d_loss.backward()  # 计算梯度
            #         d_optimizer.step()  # 更新参数
            #
            #     # 更新生成器
            #     z = torch.Tensor(torch.randn(num_img, z_dimension)).to(device)
            #     z.requires_grad = True
            #     fake_labels = generatelabels(batchsize)
            #     fake_labels.requires_grad = True
            #
            #     fake_img = G(z, fake_labels)
            #     fake_out = D(fake_img, fake_labels)
            #     g_loss = criterion(fake_out, real_label)
            #
            #     writer.add_scalar('g_loss', scale * g_loss, step)
            #
            #     g_optimizer.zero_grad()
            #     g_loss.backward()
            #     g_optimizer.step()

            if (i + 1) % 50 == 0:
                print('Epoch[{}/{}],d_loss: {:.6f},g_loss: {:.6f}'
                      'D real: {:.6f}, D fake: {:.6f}'.format(
                    epoch, epoch_num, d_loss * scale, g_loss * scale,
                    real_scores.data.mean(), fake_scores.data.mean()
                )
                )
                grid = utils.make_grid(fake_img.cpu().data)
                plt.imshow(grid.numpy().transpose((1, 2, 0)))
                plt.savefig('./img/fake_images-{}.png'.format(epoch+1))
                plt.show()
                plt.close
                    # real_images = to_img(imgs.cpu().data)
                    # save_image(real_images, './img/real_images.png', nrow=4, padding=0)

            step += 1




    # 训练完成后保存模型文件
    torch.save(G.state_dict(), './generator.pth')
    torch.save(D.state_dict(), './discriminator.pth')



def generatelabels(batchsize,real_labels =None):
    x = torch.Tensor(torch.zeros(batchsize,22)).to(device)
    if real_labels is None: #生成随机标签
        y = [np.random.randint(0, 22) for i in range(batchsize)]
        x[np.arange(batchsize), y] = 1
    else:
        x[np.arange(batchsize),real_labels] = 1
        #print("batch size:{} shape: {}" .format(batchsize, x.shape))
    return x


def to_img(x):

    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 60, 80)

    return out

