"""========Default import===="""
import argparse
import os
from numpy.ma.core import get_data
import torch.nn as nn
import torch
import wandb
import os
import torch.optim as optim
from utils.base_utils import set_random_seeds, get_accuracy, AverageMeter, WarmupCosineSchedule, WarmupLinearSchedule
from tqdm import tqdm
from easydict import EasyDict as edict
import yaml
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
""" ==========END================"""

""" =========Configurable ======="""
from models.Unet import Unet
from trainer.DDPM import DDPM
""" ===========END=========== """

def parse_args():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config",
            type=str, 
            help="Configuration file in configs.")
    parser.add_argument("--multigpu",
            type=bool, 
            help="Local rank. Necessary for using the torch.distributed.launch utility.",
            default= False)
    parser.add_argument("--resume",
            type=int,
            default= 0,
            help="Resume the last training from saved checkpoint.")
    parser.add_argument("--test", 
            type=int,
            default = 0,
            help="if test, choose True.")
    args = parser.parse_args()
    return args

def get_data_loader(cfg, args):
    if cfg.dataset in ['cifar100', 'cifar10']:
        from datasets.cifar import get_loader_cifar as get_loader
    if cfg.dataset in ['celebA']:
        from datasets.celebA import get_loader_celeba as get_loader
    if cfg.dataset in ['mnist']:
        from datasets.mnist import get_loader_mnist as get_loader

    return get_loader(cfg, args)




def main():
    args = parse_args()
    cfg = edict(yaml.safe_load(open(args.config)))
    if args.multigpu:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        if int(os.environ["RANK"]) !=0:
            cfg.wandb.active=False
    else:
        args.local_rank = -1
    if not cfg.wandb.active:
        os.environ['WANDB_SILENT'] = "true"
    if args.local_rank in [-1,0]:
        wandb.init(project = cfg.wandb.project,
                   entity = cfg.wandb.id,
                   config = dict(cfg),
                   name = f'{cfg.wandb.name}_lr:{cfg.lr}_fp16:{cfg.use_amp}',
                   group = cfg.dataset
                   )
    if args.local_rank in [-1,0]:
        print(cfg)
    # set cuda flag
    if not torch.cuda.is_available():
        print("WARNING: You have a CUDA device, so you should probably enable CUDA")
    # Set Automatic Mixed Precision
    # We need to use seeds to make sure that model initialization same
    set_random_seeds(random_seed = cfg.random_seed)
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    if args.multigpu: 
        torch.distributed.init_process_group(backend="nccl")
        # torch.distributed.init_process_group(backend="gloo")
        cfg.device = torch.device("cuda:{}".format(args.local_rank))
    else:
        cfg.device = torch.device("cuda:0")

    # torch.distributed.init_process_group(backend="gloo")

    # Encapsulate the model on the GPU assigned to the current process
    model = Unet(
            dim = cfg.img_size,
            channels= cfg.channels,
            init_dim=32,
            dim_mults = (1, 2, 4, 8)
            )
    # if custom_pre-trained model : model.load_from(np.load(<path>))
    model = model.to(cfg.device)
    if args.resume or args.test:
        #ckpt = load_ckpt(cfg.ckpt_fpath, is_best = args.test)
        ckpt = load_ckpt(cfg.ckpt_fpath)
        model.load_state_dict(ckpt['model'])
    if args.multigpu:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    trainer = DDPM(wandb = wandb, 
            args = args,
            cfg = cfg, 
            model = model, 
            get_data_loader = get_data_loader
            )

    trainer.train()

#    if args.test:
#        test(wandb, args, cfg, model, )
#    else:


if __name__ == '__main__':
    main()
