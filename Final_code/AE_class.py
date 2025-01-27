#%% Importation of the libraries
import torch
import torch.nn as nn  # Import nn to create the network layers
import torch.optim as optim  # Import optim to define the optimizer
from torch.utils.data import DataLoader, TensorDataset  # DataLoader to load and batch the data
from dataloader import DataLoader as DL
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class Autoencoder(nn.Module):
    def __init__(self, nsize=1000, latent_size=6, deepness=5):
        super(Autoencoder, self).__init__()
        
        # Calcul du facteur pour réduire la taille
        factor = (nsize / latent_size) ** (1 / deepness)  # Réduction progressive
        
        #  Encoder
        self.encoder_layers = []
        for i in range(deepness - 1):
            in_size = int(round(nsize / factor**(i)))
            out_size = int(round(nsize / factor**(i + 1)))
            self.encoder_layers.append(nn.Linear(in_size, out_size))
            self.encoder_layers.append(nn.ReLU())
        self.encoder_layers.append(nn.Linear(int(round(nsize / factor**(deepness - 1))), latent_size))
        self.encoder = nn.Sequential(*self.encoder_layers)
        
        #  Decoder
        self.decoder_layers = []
        for i in range(deepness - 1):
            in_size = int(round(latent_size * factor**(i)))
            out_size = int(round(latent_size * factor**(i + 1)))
            self.decoder_layers.append(nn.Linear(in_size, out_size))
            self.decoder_layers.append(nn.ReLU())
        self.decoder_layers.append(nn.Linear(int(round(latent_size * factor**(deepness - 1))), nsize))
        self.decoder = nn.Sequential(*self.decoder_layers)
        
    def forward(self, x):
        x= x.float()
        x = self.encoder(x)  # Compress input
        #x = (x - torch.mean(x)) / torch.std(x)
        x = self.decoder(x)  # Reconstruct input
        
        x = (x - torch.mean(x)) / torch.std(x)
        return x
    
    def encode(self, x):
        x= x.float()
        x = self.encoder(x)
        x = (x - torch.mean(x)) / torch.std(x)
        return x
    
    def decode(self, x):
        x= x.float()
        x = self.decoder(x)
        #x = (x - torch.mean(x)) / torch.std(x)
        return x
    
    def predict(self, x):
        x= x.float()
        x = self.encoder(x)
        meanx = torch.mean(x)
        stdx = torch.std(x)
        x = (x - meanx) / stdx
        x = self.decoder(x)
        x = x * stdx + meanx
        return x
