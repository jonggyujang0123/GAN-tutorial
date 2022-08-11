import torch
from utils.base_utils import set_random_seeds, get_accuracy, AverageMeter, WarmupCosineSchedule, WarmupLinearSchedule
import shutil
import os
from tqdm import tqdm
from utils.diff import linear_beta_schedule
import torchvision.utils as vutils
import wandb
import torch.optim as optim
from einops import rearrange
import torch.nn.functional as F
class DDPM():
    def __init__(
            self,
            wandb,
            args,
            cfg,
            model,
            get_data_loader,
            ):
        self.wandb = wandb
        self.model = model
        self.args = args
        self.cfg = cfg
        self.optimizer = optim.Adam(
                self.model.parameters(),
                lr = cfg.lr,
                betas = (cfg.beta, 0.999),
                )
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp) 
        self.train_loader, self.val_loader, self.test_loader = get_data_loader(cfg=cfg, args=args)

        if cfg.decay_type == "cosine":
            self.scheduler = WarmupCosineSchedule(self.optimizer, warmup_steps=cfg.warmup_steps, t_total=cfg.epochs*len(self.train_loader))
        else:
            self.scheduler = WarmupLinearSchedule(self.optimizer, warmup_steps=cfg.warmup_steps, t_total=cfg.epochs*len(self.train_loader))
        ## VAE SETTINGS
        self.timesteps = self.cfg.timesteps
        self.betas = linear_beta_schedule(timesteps= self.timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_comprod_prev = F.pad(self.alphas_cumprod[:-1], (1,0), value=1.0) # pad 1 in the left 
        self.sqrt_recip_alphas = torch.sqrt(1.0/self.alphas)

        # calculations for diffusion and others q(x_t | x_{t-1})

        self.sqrt_alphas_cumprod =torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1}|x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_comprod_prev) / (1. - self.alphas_cumprod)

    def p_losses(self, x_start, t, noise=None, loss_type = "l1"):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start, t, noise=noise)

        predicted_noise = self.model(x_noisy, t)
    
        if loss_type == "l1":
            loss =F.l1_loss(noise,predicted_noise)
        elif loss_type =="l2":
            loss=F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise,predicted_noise)
        else:
            raise NotImplementedError()
        return loss
    
    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu()) # dim= -1 (last), t is binary vector, choose the element of a with index of t. SAME FOR THE INVERSE OF ONE_HOT ENCODING
        return out.reshape(batch_size, *((1,)*(len(x_shape)-1))).to(t.device)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise  = torch.randn_like(x_start)
        sqrt_alphas_comprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_comprod_t * x_start + noise * sqrt_one_minus_alphas_cumprod_t
    

    def get_noisy_image(self, x_start, t):
        # add noise 
        x_noisy = self.q_sample(x_start, t)
        # turn back into PIL image? 
        #noisy_image = reverse_transform(x_noisy.squeeze())
        noisy_image = x_noisy.squeeze()
        return noisy_image
    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
                self.sqrt_one_minus_alphas_cumprod, t, x.shape
                )
        sqrt_recip_alphas_t = self.extract( self.sqrt_recip_alphas, t, x.shape)

        ## Eqn 11 in the paper 
        ## which estimates the mean of the noise (nosie precdiction or denoising)
        model_mean = sqrt_recip_alphas_t * (x - betas_t/sqrt_one_minus_alphas_cumprod_t * self.model(x, t))
        if t_index ==0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    # Algorithm 2 (including returning all images)
    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = next(self.model.parameters()).device()

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device= device)
        imgs = []

        for i in tqdm(reversed(range(0, self.timesteps)), desc = 'sampling loop time step', total=self.timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),i) 
            imgs.append(img.cpu().transpose(2, 3, 1))

        return imgs
    @torch.no_grad()
    def sample(self, img_size, batch_size=16, channels =3):
        return self.p_sample_loop(shape= (batch_size, channels, img_size, img_size))



    def load_ckpt(self, is_best =False):
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
        checkpoint_fpath = self.cfg.ckpt_fpath
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


    def save_ckpt(self,checkpoint, is_best=False):
        """
        Checkpoint saver
        :checkpoint_fpath : directory of the saved file
        :checkpoint : checkpoiint directory
        :return:
        """
        ckpt_path = self.cfg.ckpt_fpath+'/'+'checkpoint.pt'
        # Save the state
        if not os.path.exists(self.cfg.ckpt_fpath):
            os.makedirs(self.cfg.ckpt_fpath)
        torch.save(checkpoint, ckpt_path)
        # If it is the best copy it to another file 'model_best.pth.tar'
    #    print("Checkpoint saved successfully to '{}' at (epoch {})\n"
    #        .format(ckpt_path, checkpoint['epoch']))
        if is_best:
            ckpt_path_best = self.cfg.ckpt_fpath+'/'+'best.pt'
            print("This is the best model\n")
            shutil.copyfile(ckpt_path,
                            ckpt_path_best)



    def resume(self, is_best= False):
        ckpt = self.load_ckpt(is_best)
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        return ckpt['epoch']+1

    def train(self):
        start_epoch =0
        # We only save model who uses device "cuda:0"
        # To resume the device for the save model would be also "cuda :0"
        if self.args.resume:
            start_epoch = self.resume()
        self.model.train()
        Loss = AverageMeter()
        for epoch in range(start_epoch, self.cfg.epochs):
            tepoch = tqdm(self.train_loader,
                    disable= self.args.local_rank not in [-1,0])
            self.model.train()
            for data in tepoch:
                self.optimizer.zero_grad()
                inputs = data[0].to(self.cfg.device)
                b_size = inputs.size(0)
                # Algorithm 1 line 3 of the paper: uniformly sample $t$ for every example in the batch
                t = torch.randint(0, self.cfg.timesteps, (b_size,), device = self.cfg.device)
                with torch.cuda.amp.autocast(enabled=self.cfg.use_amp):
                    loss = self.p_losses(
                            x_start=inputs,
                            t = t,
                            loss_type="huber")
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                if self.args.local_rank in [-1,0]:
                    Loss.update(loss.mean().item(), inputs.size(0))
                    tepoch.set_description(
                            f'Epoch {epoch}: loss is {Loss.avg}, lr is {self.scheduler.get_lr()[0]:.2E}'
                            )
            self.test()
            if self.args.local_rank in [-1,0]:
                model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                ckpt = {
                        'model' : model_to_save.state_dict(),
                        'scheduler' : self.scheduler.state_dict(),
                        'optimizer' : self.optimizer.state_dict(),
                        'epoch' : epoch
                        }
                self.save_ckpt(checkpoint= ckpt)
                self.wandb.log({
                    "loss" : Loss.avg,
                    })
            if self.args.multigpu:
                torch.distributed.barrier()
    def test(self):
        self.resume(is_best=False)
        samples = self.sample(
                img_size = self.cfg.img_size, 
                batch_size = 64, 
                channels = self.cfg.channels) 
        # List of images (timesteps) with torch tensor (batch_size, height, width, channel)
        ## Grid View
        if self.args.local_rank in [-1,0]:
            img = wandb.Image(
                    vutils.make_grid(samples[-1], nrow=8, normalize= True)/2. + 0.5,
                    caption = 'generated figure for vaiorus latent vectors')
            self.wandb.log({
                "generated" : img
                })
            ## GIF view
            gif = wandb.Video(
                    rearrange(
                        torch.cat(samples),
                        '(t b1 b2) h w c -> t (b1 h) (b2 w) c', b1=8, b2=8).numpy() / 2. + 0.5,
                    fps = 4, 
                    format = "gif"
                    )
            self.wandb.log({
                "gif" : gif
                })

