"""========Default import===="""
import argparse
import os
from utils.base_utils import set_random_seeds, get_accuracy, AverageMeter, WarmupCosineSchedule, WarmupLinearSchedule
import torch.nn as nn
import torch
import wandb
import os
import torch.optim as optim
import shutil
from tqdm import tqdm
from easydict import EasyDict as edict
import yaml
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
""" ==========END================"""

""" =========Configurable ======="""
from models.VAE import VAE
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





def load_ckpt(checkpoint_fpath, is_best =False):
    """
    Latest checkpoint loader
        checkpoint_fpath : 
    :return: dict
        checkpoint{
            model,
            optimizer,
            epoch,
            scheduler}
    example :
    """
    if is_best:
        ckpt_path = checkpoint_fpath+'/'+'best.pt'
    else:
        ckpt_path = checkpoint_fpath+'/'+'checkpoint.pt'
    try:
        print(f"Loading checkpoint '{ckpt_path}'")
        checkpoint = torch.load(ckpt_path)
    except:
        print(f"No checkpoint exists from '{ckpt_path}'. Skipping...")
        print("**First time to train**")
    return checkpoint


def save_ckpt(checkpoint_fpath, checkpoint, is_best=False):
    """
    Checkpoint saver
    :checkpoint_fpath : directory of the saved file
    :checkpoint : checkpoiint directory
    :return:
    """
    ckpt_path = checkpoint_fpath+'/'+'checkpoint.pt'
    # Save the state
    if not os.path.exists(checkpoint_fpath):
        os.makedirs(checkpoint_fpath)
    torch.save(checkpoint, ckpt_path)
    # If it is the best copy it to another file 'model_best.pth.tar'
#    print("Checkpoint saved successfully to '{}' at (epoch {})\n"
#        .format(ckpt_path, checkpoint['epoch']))
    if is_best:
        ckpt_path_best = checkpoint_fpath+'/'+'best.pt'
        print("This is the best model\n")
        shutil.copyfile(ckpt_path,
                        ckpt_path_best)

# Reconstruction + KL divergence loss
def loss_function(recon_x, x, mu ,log_var):
    BCE = F.binary_cross_entropy(recon_x, torch.flatten(x,1), reduction='sum')
    KLD = -0.5 * torch.sum(1+log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


def train(wandb, args, cfg, model):
    ## Setting 
    # optmizer 
    optimizer = optim.Adam(model.parameters(), lr = cfg.lr, betas = (cfg.beta, 0.999))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
    train_loader, val_loader, test_loader = get_data_loader(cfg=cfg, args=args)
    if cfg.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=cfg.warmup_steps, t_total=cfg.epochs*len(train_loader))
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=cfg.warmup_steps, t_total=cfg.epochs*len(train_loader))
    
    # Start Training Loop
    start_epoch =0

    # We only save the model who uses device "cuda:0"
    # To resume the device for the save model woule be also "cuda:0"
    if args.resume:
        ckpt = load_ckpt(cfg.ckpt_fpath)
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']+1

        
    model.train()
    # List of Tracking results 
    Loss= AverageMeter()

    # Prepare dataset and dataloader
    for epoch in range(start_epoch, cfg.epochs):
#        if args.local_rank not in [-1,1]:
#            print("Local Rank: {}, Epoch: {}, Training ...".format(args.local_rank, epoch))

        tepoch = tqdm(train_loader, 
                      disable=args.local_rank not in [-1,0])
        model.train()
        for data in tepoch:
            ########################################
            # (1) Update Discriminator (Real data)
            ########################################
            optimizer.zero_grad()
            inputs = data[0].to(cfg.device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                recon_batch, mu, log_var = model(inputs) 
                loss = loss_function(recon_batch, inputs, mu, log_var)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if args.local_rank in [-1,0]:
                Loss.update(loss.mean().item()/inputs.size(0),inputs.size(0))
            if args.local_rank in [-1,0]:
                tepoch.set_description(f'Epoch {epoch}: loss is {Loss.avg}, lr is {scheduler.get_lr()[0]:.2E} ')
        test(wandb,args, cfg, model, test_loader)
        if args.local_rank in [-1,0]:
            model_to_save = model.module if hasattr(model, 'module') else model
            ckpt = {
                    'model': model_to_save.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch
                    }
            save_ckpt(checkpoint_fpath=cfg.ckpt_fpath, checkpoint=ckpt)
            wandb.log({
                "loss": Loss.avg,
                })
#            print("-"*75+ "\n")
#            print(f"| {epoch}-th epoch, D(x): {D_x_total.avg:2.2f}, D(G(z)): {D_G_z_total.avg:2.2f}\n")
#            print("-"*75+ "\n")
        if args.multigpu:
            torch.distributed.barrier()
#        Loss_g.reset()
#        Loss_d.reset()
#        D_x_total.reset()
#        D_G_z_total.reset()
#        tepoch.close()

def test(wandb, args, cfg, model, test_loader):
    # 1. Compute Test Loss
    # 2. wandb log 2d scatter plot
    # 3. wandb Image grid plot (fake vs real)
    # 4. wandb Image grid plot (for different basis)
    model.eval()
    Loss = AverageMeter()
    for bat_idx, data in enumerate(test_loader):
        with torch.no_grad():
            inputs = data[0].to(cfg.device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                recon_batch, mu, log_var = model(inputs) 
                loss = loss_function(recon_batch, inputs, mu, log_var)
            Loss.update(loss.mean().item(), inputs.size(0))
        if bat_idx ==0 and args.local_rank in [-1,0]:
            # 2. img_grid_plot
            img_real = vutils.make_grid(inputs, padding=2, normalize=True).cpu().numpy().transpose(1,2,0)
            img_trans = vutils.make_grid(recon_batch.view(-1, 1, cfg.img_size, cfg.img_size), padding=2, normalize=True).cpu().numpy().transpose(1,2,0)
            img_total = wandb.Image(
                    np.concatenate(
                        (img_real, img_trans),
                        axis=1),
                    caption = f'left : original, right : fake')
            wandb.log({"real vs fake" : img_total})
            # 3. scatter plot 
            label = data[1].numpy()
            mu = mu.cpu().detach().numpy()
            plt.figure(figsize=(8,6))
            plt.scatter(mu[:,0], mu[:,1], 
                    c = label, 
                    cmap = 'gist_rainbow', 
                    edgecolors='black',
                    linewidth=0.5
                    )
            plt.colorbar()
            wandb.log({"chart": wandb.Image(plt)})
            plt.close()
    # 4. wandb Image grid plot (for different latents)
    nx = ny = 20
    samples = np.mgrid[-3:3:6/nx, -3:3:6/ny].transpose(1,2,0).reshape(nx*ny,-1)
    samples = model.module.decode(torch.tensor(samples, dtype = torch.float).to(cfg.device))
    canvas = vutils.make_grid(samples.view(-1,1,cfg.img_size, cfg.img_size), nrow=20).cpu().numpy().transpose(1,2,0)
    if args.local_rank in [-1,0]:
        wandb.log({
            "Latent": wandb.Image(
                canvas,
                caption = 'generated figure for various latent vectors')
                })
            
    result = torch.tensor([Loss.sum, Loss.count]).to(cfg.device)
    if args.multigpu:
        torch.distributed.barrier()
        torch.distributed.reduce(result, op=torch.distributed.ReduceOp.SUM, dst=0)
    if args.local_rank in [-1,0]:
        wandb.log({"Loss_test" : result[0]/result[1]})


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
    if args.local_rank in [-1,0] and cfg.wandb.active:
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
    model = VAE(img_size = cfg.img_size,
            n_z = cfg.n_z)
    # if custom_pre-trained model : model.load_from(np.load(<path>))
    model = model.to(cfg.device)
    if args.resume or args.test:
        #ckpt = load_ckpt(cfg.ckpt_fpath, is_best = args.test)
        ckpt = load_ckpt(cfg.ckpt_fpath)
        model.load_state_dict(ckpt['model'])
    if args.multigpu:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)


    train(wandb, args, cfg, model)

#    if args.test:
#        test(wandb, args, cfg, model, )
#    else:


if __name__ == '__main__':
    main()
