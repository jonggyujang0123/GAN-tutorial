"""
Mnist tutorial main model
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
class VAE(nn.Module):
    def __init__(self, img_size = 28, n_z = 20):
        super().__init__()
        self.en1 = nn.Linear(img_size**2, 512)
        self.en2 = nn.Linear(512, 256)
        self.en3_1 = nn.Linear(256, n_z)
        self.en3_2 = nn.Linear(256,n_z)

        self.de1 = nn.Linear(n_z, 256)
        self.de2 = nn.Linear(256, 512)
        self.de3 = nn.Linear(512,img_size**2)

    def encode(self,x):
        h1= F.relu(self.en1(x))
        h2= F.relu(self.en2(h1))
        mean = self.en3_1(h2)
        log_var = self.en3_2(h2)
        return mean, log_var

    def decode(self, z):
        h1 = F.relu( self.de1(z))
        h2 = F.relu( self.de2(h1))
        h3 = torch.sigmoid(self.de3(h2))
        return h3
    
    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mean + std * eps

    def forward(self, x):
        mean, log_var = self.encode(torch.flatten(x,1))
        z = self.reparameterize(mean,log_var)
        return self.decode(z), mean, log_var 
