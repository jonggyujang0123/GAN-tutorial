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
""" ==========END================"""

""" =========Configurable ======="""
from models.Conv_T import Generator, Discriminator, weights_init
#os.environ["WANDB_SILENT"] = 'true'

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




def train(wandb, args, cfg, ddp_model_g, ddp_model_d):
    ## Setting 
    criterion = nn.BCELoss()
    # Create batch of latent vectors that will be used for visualization
    fixed_noise = torch.randn(cfg.test_batch_size, cfg.n_z, 1,1, device = cfg.device)

    # real/fake labels
    real_label =1.0
    fake_label= 0.0

    # optmizer 
    optimizer_d = optim.Adam(ddp_model_d.parameters(), lr = cfg.lr, betas = (cfg.beta, 0.999))
    optimizer_g = optim.Adam(ddp_model_g.parameters(), lr = cfg.lr, betas = (cfg.beta, 0.999))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
    train_loader = get_data_loader(cfg=cfg, args=args)
    if cfg.decay_type == "cosine":
        scheduler_d = WarmupCosineSchedule(optimizer_d, warmup_steps=cfg.warmup_steps, t_total=cfg.epochs*len(train_loader))
        scheduler_g = WarmupCosineSchedule(optimizer_g, warmup_steps=cfg.warmup_steps, t_total=cfg.epochs*len(train_loader))
    else:
        scheduler_d = WarmupLinearSchedule(optimizer_d, warmup_steps=cfg.warmup_steps, t_total=cfg.epochs*len(train_loader))
        scheduler_g = WarmupLinearSchedule(optimizer_g, warmup_steps=cfg.warmup_steps, t_total=cfg.epochs*len(train_loader))
    
    # Start Training Loop
    start_epoch =0

    # We only save the model who uses device "cuda:0"
    # To resume the device for the save model woule be also "cuda:0"
    if args.resume:
        ckpt = load_ckpt(cfg.ckpt_fpath)
        optimizer_d.load_state_dict(ckpt['optimizer_d'])
        optimizer_g.load_state_dict(ckpt['optimizer_g'])
        scheduler_d.load_state_dict(ckpt['scheduler_d'])
        scheduler_g.load_state_dict(ckpt['scheduler_g'])
        start_epoch = ckpt['epoch']+1

        
    ddp_model_g.train()
    ddp_model_d.train()
    # List of Tracking results 
    Loss_g= AverageMeter()
    Loss_d= AverageMeter()
    D_x_total= AverageMeter()
    D_G_z_total= AverageMeter()

    # Prepare dataset and dataloader
    for epoch in range(start_epoch, cfg.epochs):
#        if args.local_rank not in [-1,1]:
#            print("Local Rank: {}, Epoch: {}, Training ...".format(args.local_rank, epoch))

        tepoch = tqdm(train_loader, 
                      disable=args.local_rank not in [-1,0])
        for data in tepoch:
            ########################################
            # (1) Update Discriminator (Real data)
            ########################################
            optimizer_d.zero_grad()
            inputs = data[0].to(cfg.device, non_blocking=True)
            label = torch.full((inputs.size(0),), real_label, dtype=torch.float, device= cfg.device)
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                outputs = ddp_model_d(inputs).view(-1)
                err_real_d = criterion(outputs, label)
            scaler.scale(err_real_d).backward()
            scaler.step(optimizer_d)
            scaler.update()
            if args.local_rank in [-1,0]:
                D_x = outputs.detach().mean().item()
                D_x_total.update(D_x, inputs.size(0))

            #######################################
            # (2) Update Discriminator (Fake data)
            #######################################
            
            optimizer_d.zero_grad()
            latent_vector = torch.randn(inputs.size(0), cfg.n_z, 1,1, device = cfg.device)
            label.fill_(fake_label)
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                fake = ddp_model_g(latent_vector)
                outputs = ddp_model_d(fake.detach()).view(-1)
                err_fake_d = criterion(outputs, label)
            scaler.scale(err_fake_d).backward()
            scaler.step(optimizer_d)
            scaler.update()
            scheduler_d.step()
            if args.local_rank in [-1,0]:
                err_d = err_fake_d + err_real_d
                Loss_d.update(err_d.detach().mean().item(),inputs.size(0))

            ########################################
            # (3) Update Generator 
            #######################################
            
            optimizer_g.zero_grad()
            label.fill_(real_label)
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                outputs = ddp_model_d(fake).view(-1)
                err_g = criterion(outputs,label)
            scaler.scale(err_g).backward()
            scaler.step(optimizer_g)
            scaler.update()
            scheduler_g.step()

            if args.local_rank in [-1,0]:
                Loss_g.update(err_g.detach().item(),inputs.size(0))
                D_G_z = outputs.mean().item()
                D_G_z_total.update(D_G_z,inputs.size(0))
            
            ########################################
            # (4) Output of the training 
            #######################################
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                    fake_sample = ddp_model_g(fixed_noise).detach().cpu()

            if args.local_rank in [-1,0]:
                tepoch.set_description(f'Epoch {epoch}: loss_D: {Loss_d.avg:2.4f}, Loss_G: {Loss_g.avg:2.4f}, lr : {scheduler_d.get_lr()[0]:.2E}')
        if args.local_rank in [-1,0]:
            model_to_save_g = ddp_model_g.module if hasattr(ddp_model_g, 'module') else ddp_model_g
            model_to_save_d = ddp_model_d.module if hasattr(ddp_model_d, 'module') else ddp_model_d
            ckpt = {
                    'model_g': model_to_save_g.state_dict(),
                    'model_d': model_to_save_d.state_dict(),
                    'optimizer_g': optimizer_g.state_dict(),
                    'optimizer_d': optimizer_d.state_dict(),
                    'scheduler_g': scheduler_g.state_dict(),
                    'scheduler_d': scheduler_d.state_dict(),
                    'epoch': epoch
                    }
            save_ckpt(checkpoint_fpath=cfg.ckpt_fpath, checkpoint=ckpt)
            if cfg.wandb.active:
                wandb.log({
                    "loss_D": Loss_d.avg,
                    "loss_G": Loss_g.avg,
                    "D(x)" : D_x_total.avg,
                    "D(G(z))": D_G_z_total.avg,
                    })
                img = wandb.Image( vutils.make_grid(fake_sample, padding=2,normalize=True).numpy().transpose(1,2,0), caption= f"Generated @ {epoch}")
                wandb.log({"Examples": img})
            print("-"*75+ "\n")
            print(f"| {epoch}-th epoch, D(x): {D_x_total.avg:2.2f}, D(G(z)): {D_G_z_total.avg:2.2f}\n")
            print("-"*75+ "\n")
        if args.multigpu:
            torch.distributed.barrier()
#        Loss_g.reset()
#        Loss_d.reset()
#        D_x_total.reset()
#        D_G_z_total.reset()
#        tepoch.close()

def test(wandb, args, cfg, model_g):
    train_loader = get_data_loader(cfg=cfg, args= args)
    fixed_noise = torch.randn(cfg.train_batch_size, cfg.n_z, 1,1, device = cfg.device)
    model.eval()
    if args.local_rank in [-1,0]:
        with torch.no_grad():
            img_real = wandb.Image(
                    vutils.make_grid(train_loader[0].to(cfg.device)[:64], padding=5, normalize=True),
                    caption = "Real Image"
                    )
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                img = model_g(fixed_noise).detach().cpu()
            img_fake = wandb.Image(
                    vutils.make_grid(img[:64], padding=5, normalize=True),
                    caption = "fake image"
                    )
            wandb.log({"img_real": img_real, "img_fake": img_fake})
            # compute the output
#            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
#                output = model(inputs).cpu().detach()
            #example_images.append(wandb.Image(data[0], caption="Pred: {} Truth: {}".format(pred[0].detach().item(), target[0])
        #wandb.log({"Examples":example_images})
#    result  = torch.tensor([acc.sum,acc.count]).to(cfg.device)
#    if args.multigpu:
#        torch.distributed.barrier()
##        torch.distributed.all_reduce(result, op =torch.distributed.ReduceOp.SUM)
#        torch.distributed.reduce(result, op =torch.distributed.ReduceOp.SUM, dst=0)
#
#    if args.local_rank in [-1,0]:
#        print("-"*75+ "\n")
#        print(f"| Testset accuracy is {result[0].cpu().item()/result[1].cpu().item()} = {result[0].cpu().item()}/{result[1].cpu().item()}\n")
#        print("-"*75+ "\n")



def main():
    args = parse_args()
    cfg = edict(yaml.safe_load(open(args.config)))
    if args.multigpu:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        if int(os.environ["RANK"]) !=0:
            cfg.wandb.active=False
    else:
        args.local_rank = -1

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
    model_g = Generator(n_z=cfg.n_z, n_gf = cfg.n_gf, n_c = 3)
    model_d = Discriminator(n_df= cfg.n_df, n_c=3)
    # if custom_pre-trained model : model.load_from(np.load(<path>))
    model_g = model_g.to(cfg.device)
    model_d = model_d.to(cfg.device)
    if args.resume or args.test:
        #ckpt = load_ckpt(cfg.ckpt_fpath, is_best = args.test)
        ckpt = load_ckpt(cfg.ckpt_fpath)
        model_g.load_state_dict(ckpt['model_g'])
        model_d.load_state_dict(ckpt['model_d'])
    else:
        weights_init(model_g)
        weights_init(model_d)
    if args.multigpu:
        model_g = torch.nn.parallel.DistributedDataParallel(model_g, device_ids=[args.local_rank], output_device=args.local_rank)
        model_d = torch.nn.parallel.DistributedDataParallel(model_d, device_ids=[args.local_rank], output_device=args.local_rank)



    if args.test:
        test(wandb, args, cfg, model_g)
    else:
        train(wandb, args, cfg, model_g, model_d)


if __name__ == '__main__':
    main()
