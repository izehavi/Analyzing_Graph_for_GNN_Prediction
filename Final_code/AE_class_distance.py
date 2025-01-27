#%% Importation of the libraries
import torch
import torch.nn as nn  # Import nn to create the network layers



class SignalAutoEncoder(nn.Module):
    def __init__(self, nsize=1000, latent_size=14, deepness=3):
        super(SignalAutoEncoder, self).__init__()
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
        """
        Passe avant de l'autoencodeur.

        Args:
            x (torch.Tensor): Signal d'entrée.

        Returns:
            torch.Tensor: Signal reconstruit.
            torch.Tensor: Représentation latente.
        """
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z

# %%
