"""
Cifar10 Dataloader implementation
Cifar100 Dataloader implementation
"""
import logging
import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

logger = logging.getLogger()

def get_loader_cifar(cfg, args):
    transform_train = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        #transforms.RandomResizedCrop((cfg.img_size, cfg.img_size), scale= (0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.data_mean, std=cfg.data_std)
        ])

    transform_val = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        #transforms.RandomResizedCrop((cfg.img_size, cfg.img_size), scale= (0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.data_mean, std=cfg.data_std)
        ])
    transform_test = transform_val

    if args.local_rank not in [-1,0]:
        torch.distributed.barrier()

    if cfg.dataset == "cifar10":
        dataset_train = datasets.CIFAR10(root="../data",
                                         train=True,
                                         download=True,
                                         transform = transform_train,
                                         )
        dataset_val = datasets.CIFAR10(root="../data",
                                         train=False,
                                         download=True,
                                         transform = transform_val,
                                         )
        dataset_test = datasets.CIFAR10(root="../data",
                                         train=False,
                                         download=True,
                                         transform=transform_test,
                                         ) 
    elif cfg.dataset == "cifar100":
        dataset_train = datasets.CIFAR100(root="../data",
                                         train=True,
                                         download=True,
                                         transform = transform_train,
                                         )
        dataset_val = datasets.CIFAR100(root="../data",
                                         train=False,
                                         download=True,
                                         transform=transform_val,
                                         )
        dataset_test = datasets.CIFAR100(root="../data",
                                         train=False,
                                         download=True,
                                         transform = transform_test,
                                         )
    else:
        raise Exception("Please check dataset name in cfg again ('cifar10' or 'cifar100'")
    
    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(dataset_train) if args.local_rank == -1 else DistributedSampler(dataset_train)
    val_sampler = RandomSampler(dataset_val) if args.local_rank == -1 else DistributedSampler(dataset_val)
#    val_sampler = SequentialSampler(dataset_val)
    test_sampler = RandomSampler(dataset_test) if args.local_rank == -1 else DistributedSampler(dataset_test)
    train_loader = DataLoader(dataset_train,
                              sampler=train_sampler,
                              batch_size = cfg.train_batch_size,
                              num_workers = cfg.num_workers,
                              pin_memory=cfg.pin_memory)
    val_loader = DataLoader(dataset_val,
                              sampler=val_sampler,
                              batch_size = cfg.train_batch_size,
                              num_workers = cfg.num_workers,
                              pin_memory=cfg.pin_memory) 
    test_loader = DataLoader(dataset_test,
                              sampler=test_sampler,
                              batch_size = cfg.test_batch_size,
                              num_workers = cfg.num_workers,
                              pin_memory=cfg.pin_memory)
    return train_loader, val_loader, test_loader
