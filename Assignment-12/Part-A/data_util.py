# -*- coding: utf-8 -*-
"""data_util.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1slheISn0mLQa6KTA1bVBkLsPdkjDvsdz
"""
import torch
import torchvision
from torchvision import datasets
import test_transform
import train_transform


class DataProducer():

  def __init__(self):
    self.train_transform = train_transform.TrainTransform()
    self.test_transform = test_transform.TestTransform()

  def cuda_check():
      cuda_check = torch.cuda.is_available()
      print("CUDA:", cuda_check)
      device = torch.device("cuda:0" if cuda_check else "cpu")
      print(device)
      return device

  # def get_transform(hflip=True):
  #   if hflip == False:
  #     transform = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  #   else:
  #     transform = transforms.Compose([transforms.RandomHorizontalFlip(),
  #     transforms.ToTensor(),
  #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  #   return transform

  def get_dataset(self):
      trainset = datasets.CIFAR10('./data', train=True, download=True, transform=self.train_transform)
      testset = datasets.CIFAR10('./data', train=False, download=True, transform=self.test_transform)
      return trainset, testset

  def get_dataloader(self,batch_size):
    cuda = torch.cuda.is_available()
    dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=16)
    train,test=self.get_dataset()
    train_loader = torch.utils.data.DataLoader(train, **dataloader_args)
    test_loader = torch.utils.data.DataLoader(test, **dataloader_args)
    return train_loader, test_loader