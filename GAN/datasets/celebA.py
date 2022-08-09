"""
CelebA Dataloader implementation, used in DCGAN
"""
import numpy as np

import imageio

import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, TensorDataset, Dataset

def get_loader_celeba(cfg, args):
    transform_train = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.CenterCrop((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.data_mean, std=cfg.data_std)
        ])

    if args.local_rank not in [-1,0]:
        torch.distributed.barrier()

    if cfg.dataset == "celebA":
        dataset_train = datasets.CelebA(root="../data",
                                         split = 'all',
                                         download=True,
                                         transform = transform_train,
                                         )
    else:
        raise Exception("Please check dataset name in cfg again ('celebA'")
    
    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(dataset_train) if args.local_rank == -1 else DistributedSampler(dataset_train)
    train_loader = DataLoader(dataset_train,
                              sampler=train_sampler,
                              batch_size = cfg.train_batch_size,
                              num_workers = cfg.num_workers,
                              pin_memory=cfg.pin_memory
                              )
    return train_loader
